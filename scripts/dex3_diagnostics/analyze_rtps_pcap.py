#!/usr/bin/env python3
"""Offline, standard-library-only RTPS header analysis for classic PCAP files.

The analyzer never opens a socket and never imports Unitree or DDS packages.  It
reads packet headers only; it does not deserialize robot messages.  This is
enough to determine whether an RTPS writer stopped putting new sequence numbers
on the wire during a subscriber callback gap.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import ipaddress
import json
import math
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import BinaryIO, Iterable, Iterator, Sequence
import unittest


PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

DLT_EN10MB = 1
DLT_RAW = 101
DLT_LINUX_SLL = 113
DLT_LINUX_SLL2 = 276
SUPPORTED_LINKTYPES = {DLT_EN10MB, DLT_RAW, DLT_LINUX_SLL, DLT_LINUX_SLL2}

ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
VLAN_ETHERTYPES = {0x8100, 0x88A8, 0x9100}
IPPROTO_UDP = 17

RTPS_DATA = 0x15
RTPS_DATA_FRAG = 0x16
RTPS_ACKNACK = 0x06
RTPS_HEARTBEAT = 0x07
RTPS_GAP = 0x08
RTPS_INFO_TS = 0x09
RTPS_INFO_SRC = 0x0C
RTPS_INFO_DST = 0x0E
RTPS_NACK_FRAG = 0x12
RTPS_HEARTBEAT_FRAG = 0x13

DEFAULT_MAX_RECORD_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_EVENTS = 1_000_000
GAP_THRESHOLDS_NS = (75_000_000, 250_000_000, 900_000_000)


class PcapError(ValueError):
    """Raised when a capture is malformed or outside the supported subset."""


@dataclass(frozen=True)
class PcapHeader:
    endian: str
    timestamp_scale_ns: int
    linktype: int
    snaplen: int


@dataclass(frozen=True)
class CapturedPacket:
    index: int
    capture_ns: int
    captured_length: int
    original_length: int
    data: bytes


@dataclass(frozen=True)
class UdpDatagram:
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    payload: bytes
    truncated: bool
    fragmented: bool


@dataclass(frozen=True)
class RtpsEvent:
    packet_index: int
    submessage_offset: int
    capture_ns: int
    source_ip: str
    source_port: int
    destination_ip: str
    destination_port: int
    protocol_version: str
    vendor_id: str
    sender_guid_prefix: str
    writer_entity: str
    writer_guid: str
    kind: str
    sequence: int | None = None
    last_sequence: int | None = None
    fragment_start: int | None = None
    fragments_in_submessage: int | None = None
    fragment_size: int | None = None
    sample_size: int | None = None
    source_timestamp_ns: int | None = None
    packet_truncated: bool = False


@dataclass
class ParseStats:
    packets: int = 0
    udp_datagrams: int = 0
    rtps_datagrams: int = 0
    truncated_packets: int = 0
    skipped_non_udp: int = 0
    skipped_ip_fragments: int = 0
    malformed_network: int = 0
    malformed_rtps: int = 0
    capped_events: bool = False
    rtps_submessages: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class WriterSummary:
    writer_guid: str
    source_ip: str
    source_port: int
    vendor_id: str
    first_capture_ns: int
    last_capture_ns: int
    unique_sequences: int
    data_submessages: int
    data_frag_submessages: int
    rate_hz: float
    median_gap_ns: int
    p95_gap_ns: int
    p99_gap_ns: int
    maximum_gap_ns: int
    gaps_over_75ms: int
    gaps_over_250ms: int
    gaps_over_900ms: int
    forward_missing_sequences: int
    duplicate_sequences: int
    reordered_sequences: int
    incomplete_fragment_samples: int


@dataclass(frozen=True)
class CallbackRow:
    stream: str
    topic: str
    receive_wall_ns: int
    source_timestamp_ns: int | None


@dataclass(frozen=True)
class WriterTopicMapping:
    writer_guid: str
    stream: str
    topic: str
    matches: int
    unambiguous_writer_timestamp_matches: int
    ambiguous_timestamp_matches: int
    confidence: float
    accepted: bool
    evidence: str


@dataclass(frozen=True)
class ProbeSummaryEvidence:
    trace_every: int
    trace_dropped: int
    stream_by_topic: dict[str, str]
    threshold_ns_by_topic: dict[str, int]
    gaps_by_topic: dict[str, tuple[dict[str, object], ...]]


def _read_exact(handle: BinaryIO, size: int, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise PcapError(f"truncated {context}: expected {size} bytes, got {len(data)}")
    return data


def read_pcap_header(handle: BinaryIO) -> PcapHeader:
    raw = _read_exact(handle, 24, "PCAP global header")
    magic = raw[:4]
    if magic == PCAPNG_MAGIC:
        raise PcapError("PCAPNG is not supported; capture classic PCAP instead")
    try:
        endian, timestamp_scale_ns = PCAP_MAGIC[magic]
    except KeyError as error:
        raise PcapError(f"unknown PCAP magic {magic.hex()}") from error

    major, minor = struct.unpack_from(f"{endian}HH", raw, 4)
    if (major, minor) != (2, 4):
        raise PcapError(f"unsupported PCAP version {major}.{minor}; expected 2.4")
    snaplen, raw_linktype = struct.unpack_from(f"{endian}II", raw, 16)
    linktype = raw_linktype & 0xFFFF
    if snaplen == 0:
        raise PcapError("invalid PCAP snaplen 0")
    if linktype not in SUPPORTED_LINKTYPES:
        raise PcapError(f"unsupported PCAP linktype {linktype}")
    return PcapHeader(endian, timestamp_scale_ns, linktype, snaplen)


def iter_pcap_packets(
    handle: BinaryIO,
    header: PcapHeader,
    *,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> Iterator[CapturedPacket]:
    index = 0
    while True:
        raw_header = handle.read(16)
        if not raw_header:
            return
        if len(raw_header) != 16:
            raise PcapError(
                f"truncated PCAP record header at packet {index}: got {len(raw_header)} bytes"
            )
        seconds, fraction, included, original = struct.unpack(f"{header.endian}IIII", raw_header)
        fraction_limit = 1_000_000 if header.timestamp_scale_ns == 1_000 else 1_000_000_000
        if fraction >= fraction_limit:
            raise PcapError(f"invalid timestamp fraction {fraction} at packet {index}")
        if included > max_record_bytes:
            raise PcapError(
                f"packet {index} captured length {included} exceeds limit {max_record_bytes}"
            )
        if included > header.snaplen:
            raise PcapError(
                f"packet {index} captured length {included} exceeds snaplen {header.snaplen}"
            )
        if included > original:
            raise PcapError(
                f"packet {index} captured length {included} exceeds original length {original}"
            )
        data = _read_exact(handle, included, f"PCAP packet {index}")
        capture_ns = seconds * 1_000_000_000 + fraction * header.timestamp_scale_ns
        yield CapturedPacket(index, capture_ns, included, original, data)
        index += 1


def open_classic_pcap(
    path: Path, *, max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES
) -> tuple[PcapHeader, Iterator[CapturedPacket], BinaryIO]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise PcapError(f"capture must be a regular, non-symlink file: {path}")
    handle = path.open("rb")
    try:
        header = read_pcap_header(handle)
    except Exception:
        handle.close()
        raise
    return header, iter_pcap_packets(handle, header, max_record_bytes=max_record_bytes), handle


def _u16_be(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _network_payload(frame: bytes, linktype: int) -> tuple[int, int]:
    if linktype == DLT_EN10MB:
        if len(frame) < 14:
            raise PcapError("truncated Ethernet header")
        ether_type = _u16_be(frame, 12)
        offset = 14
        while ether_type in VLAN_ETHERTYPES:
            if len(frame) < offset + 4:
                raise PcapError("truncated VLAN header")
            ether_type = _u16_be(frame, offset + 2)
            offset += 4
        return ether_type, offset
    if linktype == DLT_LINUX_SLL:
        if len(frame) < 16:
            raise PcapError("truncated Linux cooked v1 header")
        return _u16_be(frame, 14), 16
    if linktype == DLT_LINUX_SLL2:
        if len(frame) < 20:
            raise PcapError("truncated Linux cooked v2 header")
        return _u16_be(frame, 0), 20
    if linktype == DLT_RAW:
        if not frame:
            raise PcapError("empty raw-IP frame")
        version = frame[0] >> 4
        if version == 4:
            return ETHERTYPE_IPV4, 0
        if version == 6:
            return ETHERTYPE_IPV6, 0
        raise PcapError(f"unknown raw-IP version {version}")
    raise PcapError(f"unsupported linktype {linktype}")


def _parse_udp(
    packet: bytes,
    offset: int,
    captured_end: int,
    source_ip: str,
    destination_ip: str,
    *,
    fragmented: bool,
) -> UdpDatagram:
    if captured_end < offset + 8:
        raise PcapError("truncated UDP header")
    source_port, destination_port, udp_length = struct.unpack_from(">HHH", packet, offset)
    if udp_length < 8:
        raise PcapError(f"invalid UDP length {udp_length}")
    expected_end = offset + udp_length
    actual_end = min(captured_end, expected_end)
    return UdpDatagram(
        source_ip=source_ip,
        destination_ip=destination_ip,
        source_port=source_port,
        destination_port=destination_port,
        payload=packet[offset + 8 : actual_end],
        truncated=actual_end < expected_end,
        fragmented=fragmented,
    )


def _parse_ipv4(frame: bytes, offset: int) -> UdpDatagram | None:
    if len(frame) < offset + 20:
        raise PcapError("truncated IPv4 header")
    first = frame[offset]
    if first >> 4 != 4:
        raise PcapError("EtherType says IPv4 but header version differs")
    ihl = (first & 0x0F) * 4
    if ihl < 20 or len(frame) < offset + ihl:
        raise PcapError(f"invalid or truncated IPv4 IHL {ihl}")
    total_length = _u16_be(frame, offset + 2)
    if total_length < ihl:
        raise PcapError(f"invalid IPv4 total length {total_length}")
    fragment_field = _u16_be(frame, offset + 6)
    fragment_offset = fragment_field & 0x1FFF
    more_fragments = bool(fragment_field & 0x2000)
    if fragment_offset != 0:
        return None
    if frame[offset + 9] != IPPROTO_UDP:
        return None
    source_ip = str(ipaddress.IPv4Address(frame[offset + 12 : offset + 16]))
    destination_ip = str(ipaddress.IPv4Address(frame[offset + 16 : offset + 20]))
    captured_end = min(len(frame), offset + total_length)
    return _parse_udp(
        frame,
        offset + ihl,
        captured_end,
        source_ip,
        destination_ip,
        fragmented=more_fragments,
    )


def _parse_ipv6(frame: bytes, offset: int) -> UdpDatagram | None:
    if len(frame) < offset + 40:
        raise PcapError("truncated IPv6 header")
    if frame[offset] >> 4 != 6:
        raise PcapError("EtherType says IPv6 but header version differs")
    payload_length = _u16_be(frame, offset + 4)
    next_header = frame[offset + 6]
    cursor = offset + 40
    captured_end = min(len(frame), cursor + payload_length)
    source_ip = str(ipaddress.IPv6Address(frame[offset + 8 : offset + 24]))
    destination_ip = str(ipaddress.IPv6Address(frame[offset + 24 : offset + 40]))
    fragmented = False

    for _ in range(8):
        if next_header == IPPROTO_UDP:
            return _parse_udp(
                frame,
                cursor,
                captured_end,
                source_ip,
                destination_ip,
                fragmented=fragmented,
            )
        if next_header in {0, 43, 60}:  # hop-by-hop, routing, destination options
            if captured_end < cursor + 2:
                raise PcapError("truncated IPv6 extension header")
            following = frame[cursor]
            length = (frame[cursor + 1] + 1) * 8
        elif next_header == 51:  # Authentication Header
            if captured_end < cursor + 2:
                raise PcapError("truncated IPv6 AH header")
            following = frame[cursor]
            length = (frame[cursor + 1] + 2) * 4
        elif next_header == 44:  # fragment header
            if captured_end < cursor + 8:
                raise PcapError("truncated IPv6 fragment header")
            following = frame[cursor]
            fragment_field = _u16_be(frame, cursor + 2)
            if (fragment_field >> 3) & 0x1FFF:
                return None
            fragmented = True
            length = 8
        else:
            return None
        if length <= 0 or captured_end < cursor + length:
            raise PcapError("truncated IPv6 extension body")
        cursor += length
        next_header = following
    raise PcapError("too many IPv6 extension headers")


def parse_udp_datagram(frame: bytes, linktype: int) -> UdpDatagram | None:
    ether_type, offset = _network_payload(frame, linktype)
    if ether_type == ETHERTYPE_IPV4:
        return _parse_ipv4(frame, offset)
    if ether_type == ETHERTYPE_IPV6:
        return _parse_ipv6(frame, offset)
    return None


def _unpack(endian: str, fmt: str, data: bytes, offset: int) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise PcapError("truncated RTPS submessage field")
    return struct.unpack_from(f"{endian}{fmt}", data, offset)


def _sequence_number(endian: str, data: bytes, offset: int) -> int:
    high, low = _unpack(endian, "iI", data, offset)
    return high * (1 << 32) + low


def _guid(prefix: bytes | None, entity: bytes) -> str:
    if prefix is None:
        return f"unknown.{entity.hex()}"
    return f"{prefix.hex()}.{entity.hex()}"


def parse_rtps_events(packet: CapturedPacket, udp: UdpDatagram) -> list[RtpsEvent]:
    payload = udp.payload
    if len(payload) < 20 or payload[:4] != b"RTPS":
        return []
    version = f"{payload[4]}.{payload[5]}"
    vendor = payload[6:8].hex()
    sender_prefix = payload[8:20]
    current_source_prefix = sender_prefix
    current_destination_prefix: bytes | None = None
    source_timestamp_ns: int | None = None
    cursor = 20
    events: list[RtpsEvent] = []

    while cursor < len(payload):
        if len(payload) - cursor < 4:
            raise PcapError("truncated RTPS submessage header")
        submessage_id = payload[cursor]
        flags = payload[cursor + 1]
        endian = "<" if flags & 0x01 else ">"
        declared_length = struct.unpack_from(f"{endian}H", payload, cursor + 2)[0]
        content_start = cursor + 4
        declared_end = len(payload) if declared_length == 0 else content_start + declared_length
        available_end = min(declared_end, len(payload))
        content = payload[content_start:available_end]
        truncated_submessage = declared_end > len(payload)

        if submessage_id == RTPS_INFO_TS:
            if flags & 0x02:
                source_timestamp_ns = None
            elif len(content) >= 8:
                seconds, fraction = struct.unpack_from(f"{endian}iI", content)
                source_timestamp_ns = seconds * 1_000_000_000 + (
                    fraction * 1_000_000_000 // (1 << 32)
                )
        elif submessage_id == RTPS_INFO_SRC and len(content) >= 20:
            current_source_prefix = content[8:20]
        elif submessage_id == RTPS_INFO_DST and len(content) >= 12:
            current_destination_prefix = content[:12]
        else:
            event: RtpsEvent | None = None
            packet_truncated = udp.truncated or truncated_submessage
            common = {
                "packet_index": packet.index,
                "submessage_offset": cursor,
                "capture_ns": packet.capture_ns,
                "source_ip": udp.source_ip,
                "source_port": udp.source_port,
                "destination_ip": udp.destination_ip,
                "destination_port": udp.destination_port,
                "protocol_version": version,
                "vendor_id": vendor,
                "sender_guid_prefix": sender_prefix.hex(),
                "source_timestamp_ns": source_timestamp_ns,
                "packet_truncated": packet_truncated,
            }
            if submessage_id == RTPS_DATA and len(content) >= 20:
                entity = content[8:12]
                event = RtpsEvent(
                    **common,
                    writer_entity=entity.hex(),
                    writer_guid=_guid(current_source_prefix, entity),
                    kind="DATA",
                    sequence=_sequence_number(endian, content, 12),
                )
            elif submessage_id == RTPS_DATA_FRAG and len(content) >= 32:
                entity = content[8:12]
                fragment_start, fragments_in, fragment_size, sample_size = _unpack(
                    endian, "IHHI", content, 20
                )
                event = RtpsEvent(
                    **common,
                    writer_entity=entity.hex(),
                    writer_guid=_guid(current_source_prefix, entity),
                    kind="DATA_FRAG",
                    sequence=_sequence_number(endian, content, 12),
                    fragment_start=fragment_start,
                    fragments_in_submessage=fragments_in,
                    fragment_size=fragment_size,
                    sample_size=sample_size,
                )
            elif submessage_id in {RTPS_HEARTBEAT, RTPS_GAP} and len(content) >= 12:
                entity = content[4:8]
                sequence = _sequence_number(endian, content, 8)
                last_sequence = None
                if submessage_id == RTPS_HEARTBEAT and len(content) >= 24:
                    last_sequence = _sequence_number(endian, content, 16)
                event = RtpsEvent(
                    **common,
                    writer_entity=entity.hex(),
                    writer_guid=_guid(current_source_prefix, entity),
                    kind="HEARTBEAT" if submessage_id == RTPS_HEARTBEAT else "GAP",
                    sequence=sequence,
                    last_sequence=last_sequence,
                )
            elif submessage_id in {RTPS_ACKNACK, RTPS_NACK_FRAG} and len(content) >= 8:
                entity = content[4:8]
                sequence = None
                if submessage_id == RTPS_NACK_FRAG and len(content) >= 16:
                    sequence = _sequence_number(endian, content, 8)
                event = RtpsEvent(
                    **common,
                    writer_entity=entity.hex(),
                    writer_guid=_guid(current_destination_prefix, entity),
                    kind="ACKNACK" if submessage_id == RTPS_ACKNACK else "NACK_FRAG",
                    sequence=sequence,
                )
            elif submessage_id == RTPS_HEARTBEAT_FRAG and len(content) >= 16:
                entity = content[4:8]
                event = RtpsEvent(
                    **common,
                    writer_entity=entity.hex(),
                    writer_guid=_guid(current_source_prefix, entity),
                    kind="HEARTBEAT_FRAG",
                    sequence=_sequence_number(endian, content, 8),
                )
            if event is not None:
                events.append(event)

        if declared_length == 0 or truncated_submessage:
            break
        cursor = declared_end
    return events


def analyze_capture(
    path: Path,
    *,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> tuple[PcapHeader, list[RtpsEvent], ParseStats]:
    if max_record_bytes <= 0 or max_events <= 0:
        raise ValueError("record and event limits must be positive")
    header, packets, handle = open_classic_pcap(path, max_record_bytes=max_record_bytes)
    events: list[RtpsEvent] = []
    stats = ParseStats()
    try:
        for packet in packets:
            stats.packets += 1
            if packet.captured_length < packet.original_length:
                stats.truncated_packets += 1
            try:
                udp = parse_udp_datagram(packet.data, header.linktype)
            except PcapError:
                stats.malformed_network += 1
                continue
            if udp is None:
                stats.skipped_non_udp += 1
                continue
            stats.udp_datagrams += 1
            if udp.fragmented:
                stats.skipped_ip_fragments += 1
            if len(udp.payload) < 4 or udp.payload[:4] != b"RTPS":
                continue
            stats.rtps_datagrams += 1
            try:
                packet_events = parse_rtps_events(packet, udp)
            except PcapError:
                stats.malformed_rtps += 1
                continue
            for event in packet_events:
                stats.rtps_submessages[event.kind] += 1
                if len(events) >= max_events:
                    stats.capped_events = True
                    return header, events, stats
                events.append(event)
    finally:
        handle.close()
    return header, events, stats


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_writers(events: Iterable[RtpsEvent]) -> list[WriterSummary]:
    grouped: dict[str, list[RtpsEvent]] = defaultdict(list)
    for event in events:
        if event.kind in {"DATA", "DATA_FRAG"} and event.sequence is not None:
            grouped[event.writer_guid].append(event)

    summaries: list[WriterSummary] = []
    for writer_guid, writer_events in grouped.items():
        writer_events.sort(
            key=lambda item: (item.capture_ns, item.packet_index, item.submessage_offset)
        )
        first_by_sequence: dict[int, RtpsEvent] = {}
        fragment_coverage: dict[int, set[int]] = defaultdict(set)
        fragment_totals: dict[int, int] = {}
        complete_data_sequences: set[int] = set()
        duplicate_sequences = 0
        reordered_sequences = 0
        maximum_sequence: int | None = None

        for event in writer_events:
            sequence = event.sequence
            assert sequence is not None
            if sequence not in first_by_sequence:
                first_by_sequence[sequence] = event
                if maximum_sequence is not None:
                    if sequence < maximum_sequence:
                        reordered_sequences += 1
                maximum_sequence = (
                    sequence if maximum_sequence is None else max(maximum_sequence, sequence)
                )

            if event.kind == "DATA":
                if sequence in complete_data_sequences:
                    duplicate_sequences += 1
                complete_data_sequences.add(sequence)
            elif event.kind == "DATA_FRAG":
                if (
                    event.fragment_start is not None
                    and event.fragments_in_submessage is not None
                    and event.fragment_size
                    and event.sample_size is not None
                    and event.fragment_start > 0
                ):
                    total = math.ceil(event.sample_size / event.fragment_size)
                    fragment_totals[sequence] = max(fragment_totals.get(sequence, 0), total)
                    stop = min(total + 1, event.fragment_start + event.fragments_in_submessage)
                    observed_fragments = set(range(event.fragment_start, stop))
                    # Different fragments of one sample are expected, not duplicate
                    # sequences. Count only a DATA_FRAG carrying no new fragment.
                    if observed_fragments and observed_fragments.issubset(
                        fragment_coverage[sequence]
                    ):
                        duplicate_sequences += 1
                    fragment_coverage[sequence].update(observed_fragments)

        first_events = sorted(
            first_by_sequence.values(),
            key=lambda item: (item.capture_ns, item.packet_index, item.submessage_offset),
        )
        arrival_times = [event.capture_ns for event in first_events]
        sorted_sequences = sorted(first_by_sequence)
        forward_missing = sum(
            max(0, later - earlier - 1)
            for earlier, later in zip(sorted_sequences, sorted_sequences[1:])
        )
        gaps = [later - earlier for earlier, later in zip(arrival_times, arrival_times[1:])]
        duration_ns = arrival_times[-1] - arrival_times[0] if len(arrival_times) > 1 else 0
        rate_hz = (len(arrival_times) - 1) * 1e9 / duration_ns if duration_ns > 0 else 0.0
        incomplete = sum(
            1
            for sequence, total in fragment_totals.items()
            if len(fragment_coverage.get(sequence, set())) < total
        )
        exemplar = first_events[0]
        summaries.append(
            WriterSummary(
                writer_guid=writer_guid,
                source_ip=exemplar.source_ip,
                source_port=exemplar.source_port,
                vendor_id=exemplar.vendor_id,
                first_capture_ns=arrival_times[0],
                last_capture_ns=arrival_times[-1],
                unique_sequences=len(first_events),
                data_submessages=sum(event.kind == "DATA" for event in writer_events),
                data_frag_submessages=sum(event.kind == "DATA_FRAG" for event in writer_events),
                rate_hz=rate_hz,
                median_gap_ns=nearest_rank(gaps, 0.50),
                p95_gap_ns=nearest_rank(gaps, 0.95),
                p99_gap_ns=nearest_rank(gaps, 0.99),
                maximum_gap_ns=max(gaps, default=0),
                gaps_over_75ms=sum(gap > GAP_THRESHOLDS_NS[0] for gap in gaps),
                gaps_over_250ms=sum(gap > GAP_THRESHOLDS_NS[1] for gap in gaps),
                gaps_over_900ms=sum(gap > GAP_THRESHOLDS_NS[2] for gap in gaps),
                forward_missing_sequences=forward_missing,
                duplicate_sequences=duplicate_sequences,
                reordered_sequences=reordered_sequences,
                incomplete_fragment_samples=incomplete,
            )
        )
    return sorted(summaries, key=lambda item: item.writer_guid)


def _integer_field(row: dict[str, str], candidates: Sequence[str]) -> int | None:
    for name in candidates:
        value = row.get(name)
        if value not in {None, ""}:
            try:
                return int(value)
            except ValueError as error:
                raise PcapError(
                    f"callback CSV field {name!r} is not an integer: {value!r}"
                ) from error
    return None


def read_callbacks(path: Path) -> list[CallbackRow]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise PcapError(f"callback CSV must be a regular, non-symlink file: {path}")
    callbacks: list[CallbackRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PcapError("callback CSV has no header")
        for index, row in enumerate(reader, start=2):
            receive_ns = _integer_field(
                row, ("receive_wall_ns", "received_utc_ns", "callback_utc_ns", "wall_time_ns")
            )
            if receive_ns is None:
                raise PcapError(
                    "callback CSV needs receive_wall_ns (or received_utc_ns/callback_utc_ns/wall_time_ns)"
                )
            source_ns = _integer_field(row, ("source_timestamp_ns", "source_utc_ns"))
            stream = row.get("stream") or row.get("side") or row.get("name") or ""
            topic = row.get("topic") or ""
            if not stream or not topic:
                raise PcapError(f"callback CSV row {index} needs explicit stream and topic")
            accepted = row.get("accepted", "true").strip().lower()
            if accepted not in {"0", "1", "false", "true", "no", "yes"}:
                raise PcapError(f"callback CSV row {index} has invalid accepted value {accepted!r}")
            if accepted in {"0", "false", "no"}:
                continue
            callbacks.append(CallbackRow(stream, topic, receive_ns, source_ns))
    return callbacks


def read_probe_summary(path: Path, callbacks: Sequence[CallbackRow]) -> ProbeSummaryEvidence:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise PcapError(f"probe summary must be a regular, non-symlink file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            summary = json.load(handle)
        except json.JSONDecodeError as error:
            raise PcapError(f"invalid probe summary JSON: {error}") from error
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise PcapError("probe summary must use schema_version 1")
    if "close_error" not in summary:
        raise PcapError("probe summary has no close_error integrity field")
    if summary["close_error"] is not None:
        raise PcapError(f"probe summary reports a close error: {summary['close_error']!r}")
    trace = summary.get("trace")
    if not isinstance(trace, dict) or trace.get("enabled") is not True:
        raise PcapError("probe summary says callback tracing was not enabled")
    trace_every = trace.get("every_nth_callback")
    trace_dropped = trace.get("dropped_queue_rows")
    if not isinstance(trace_every, int) or isinstance(trace_every, bool) or trace_every < 1:
        raise PcapError("probe summary has an invalid trace every_nth_callback")
    if not isinstance(trace_dropped, int) or isinstance(trace_dropped, bool) or trace_dropped != 0:
        raise PcapError(
            f"callback trace dropped {trace_dropped!r} rows; writer/topic mapping is not trusted"
        )

    raw_streams = summary.get("streams")
    if not isinstance(raw_streams, dict) or not raw_streams:
        raise PcapError("probe summary has no stream evidence")
    stream_by_topic: dict[str, str] = {}
    threshold_ns_by_topic: dict[str, int] = {}
    gaps_by_topic: dict[str, tuple[dict[str, object], ...]] = {}
    for stream, raw in raw_streams.items():
        if not isinstance(stream, str) or not isinstance(raw, dict):
            raise PcapError("probe summary stream entry is malformed")
        topic = raw.get("topic")
        threshold_s = raw.get("gap_threshold_s")
        raw_gaps = raw.get("gaps")
        if not isinstance(topic, str) or not topic:
            raise PcapError(f"probe summary stream {stream!r} has no topic")
        if (
            not isinstance(threshold_s, (int, float))
            or isinstance(threshold_s, bool)
            or not math.isfinite(threshold_s)
            or threshold_s <= 0
        ):
            raise PcapError(f"probe summary stream {stream!r} has an invalid threshold")
        if not isinstance(raw_gaps, list):
            raise PcapError(f"probe summary stream {stream!r} has no gap list")
        if topic in stream_by_topic:
            raise PcapError(f"probe summary repeats topic {topic!r}")
        threshold_ns = round(float(threshold_s) * 1e9)
        normalized_gaps: list[dict[str, object]] = []
        for gap in raw_gaps:
            if not isinstance(gap, dict):
                raise PcapError(f"probe summary stream {stream!r} has a malformed gap")
            start_ns = gap.get("start_wall_ns")
            end_ns = gap.get("end_wall_ns")
            recovered = gap.get("recovered")
            if (
                not isinstance(start_ns, int)
                or isinstance(start_ns, bool)
                or not isinstance(end_ns, int)
                or isinstance(end_ns, bool)
                or end_ns <= start_ns
                or not isinstance(recovered, bool)
            ):
                raise PcapError(f"probe summary stream {stream!r} has invalid gap bounds")
            if end_ns - start_ns <= threshold_ns:
                raise PcapError(
                    f"probe summary gap for {stream!r} does not exceed its declared threshold"
                )
            normalized_gaps.append(
                {
                    "start_wall_ns": start_ns,
                    "end_wall_ns": end_ns,
                    "recovered": recovered,
                }
            )
        stream_by_topic[topic] = stream
        threshold_ns_by_topic[topic] = threshold_ns
        gaps_by_topic[topic] = tuple(normalized_gaps)

    callback_topics = {callback.topic for callback in callbacks}
    unknown_topics = sorted(callback_topics - stream_by_topic.keys())
    if unknown_topics:
        raise PcapError(
            "callback trace contains topics absent from probe summary: " + ", ".join(unknown_topics)
        )
    for callback in callbacks:
        expected_stream = stream_by_topic[callback.topic]
        if callback.stream and callback.stream != expected_stream:
            raise PcapError(
                f"callback stream/topic mismatch: {callback.stream!r} vs {callback.topic!r}"
            )
    if trace_every == 1:
        accepted_by_stream = Counter(callback.stream for callback in callbacks)
        for stream, raw in raw_streams.items():
            accepted_count = raw.get("accepted_count")
            if (
                not isinstance(accepted_count, int)
                or isinstance(accepted_count, bool)
                or accepted_count < 0
            ):
                raise PcapError(f"full-trace summary has invalid accepted_count for {stream!r}")
            if accepted_by_stream[stream] != accepted_count:
                raise PcapError(
                    f"full callback trace count for {stream!r} is {accepted_by_stream[stream]}, "
                    f"summary says {accepted_count}"
                )
    return ProbeSummaryEvidence(
        trace_every=trace_every,
        trace_dropped=trace_dropped,
        stream_by_topic=stream_by_topic,
        threshold_ns_by_topic=threshold_ns_by_topic,
        gaps_by_topic=gaps_by_topic,
    )


def map_writers_to_topics(
    events: Sequence[RtpsEvent],
    callbacks: Sequence[CallbackRow],
    *,
    tolerance_ns: int = 1_000,
    minimum_matches: int = 5,
    minimum_confidence: float = 0.95,
) -> list[WriterTopicMapping]:
    timestamped = sorted(
        (event.source_timestamp_ns, event.writer_guid)
        for event in events
        if event.kind in {"DATA", "DATA_FRAG"} and event.source_timestamp_ns is not None
    )
    timestamps = [item[0] for item in timestamped]
    matches: Counter[tuple[str, str, str]] = Counter()
    unambiguous_totals: Counter[str] = Counter()
    ambiguous_totals: Counter[str] = Counter()
    observed_writers: set[str] = set()
    for callback in callbacks:
        if callback.source_timestamp_ns is None:
            continue
        low = bisect.bisect_left(timestamps, callback.source_timestamp_ns - tolerance_ns)
        high = bisect.bisect_right(timestamps, callback.source_timestamp_ns + tolerance_ns)
        matched_writers = {timestamped[index][1] for index in range(low, high)}
        observed_writers.update(matched_writers)
        if len(matched_writers) != 1:
            for writer_guid in matched_writers:
                ambiguous_totals[writer_guid] += 1
            continue
        writer_guid = next(iter(matched_writers))
        matches[(writer_guid, callback.stream, callback.topic)] += 1
        unambiguous_totals[writer_guid] += 1

    mappings: list[WriterTopicMapping] = []
    for writer_guid in sorted(observed_writers):
        candidates = [item for item in matches if item[0] == writer_guid]
        if not candidates:
            mappings.append(
                WriterTopicMapping(
                    writer_guid=writer_guid,
                    stream="",
                    topic="",
                    matches=0,
                    unambiguous_writer_timestamp_matches=0,
                    ambiguous_timestamp_matches=ambiguous_totals[writer_guid],
                    confidence=0.0,
                    accepted=False,
                    evidence="ambiguous_timestamp_matches_only",
                )
            )
            continue
        best = max(candidates, key=lambda item: (matches[item], item[2], item[1]))
        count = matches[best]
        unambiguous_total = unambiguous_totals[writer_guid]
        evidence_total = unambiguous_total + ambiguous_totals[writer_guid]
        confidence = count / evidence_total
        accepted = count >= minimum_matches and confidence >= minimum_confidence
        if count < minimum_matches:
            evidence = "insufficient_unambiguous_timestamp_matches"
        elif confidence < minimum_confidence:
            evidence = "ambiguous_topic_assignment"
        else:
            evidence = (
                "timestamp_mapping_accepted_ambiguous_matches_excluded"
                if ambiguous_totals[writer_guid]
                else "timestamp_mapping_accepted"
            )
        mappings.append(
            WriterTopicMapping(
                writer_guid=writer_guid,
                stream=best[1],
                topic=best[2],
                matches=count,
                unambiguous_writer_timestamp_matches=unambiguous_total,
                ambiguous_timestamp_matches=ambiguous_totals[writer_guid],
                confidence=confidence,
                accepted=accepted,
                evidence=evidence,
            )
        )
    return mappings


def correlate_probe_gaps(
    events: Sequence[RtpsEvent],
    mappings: Sequence[WriterTopicMapping],
    probe_summary: ProbeSummaryEvidence,
    *,
    capture_kernel_drops: int = 0,
) -> list[dict[str, object]]:
    writers_by_topic: dict[str, list[str]] = defaultdict(list)
    for mapping in mappings:
        if mapping.accepted:
            writers_by_topic[mapping.topic].append(mapping.writer_guid)
    wire_times: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for event in sorted(events, key=lambda item: (item.capture_ns, item.packet_index)):
        if event.kind not in {"DATA", "DATA_FRAG"} or event.sequence is None:
            continue
        key = (event.writer_guid, event.sequence)
        if key not in seen:
            seen.add(key)
            wire_times[event.writer_guid].append(event.capture_ns)

    result: list[dict[str, object]] = []
    for topic, gaps in sorted(probe_summary.gaps_by_topic.items()):
        candidate_writers = writers_by_topic.get(topic, [])
        writer_guid = candidate_writers[0] if len(candidate_writers) == 1 else ""
        times = wire_times.get(writer_guid, [])
        threshold_ns = probe_summary.threshold_ns_by_topic[topic]
        for gap in gaps:
            start_ns = int(gap["start_wall_ns"])
            end_ns = int(gap["end_wall_ns"])
            left = bisect.bisect_right(times, start_ns)
            right = bisect.bisect_left(times, end_ns)
            wire_inside = max(0, right - left)
            local_wire_gaps: list[int] = []
            start_index = max(0, left - 1)
            stop_index = min(len(times) - 1, right + 1)
            for index in range(start_index, stop_index):
                earlier, later = times[index], times[index + 1]
                if earlier <= end_ns and later >= start_ns:
                    local_wire_gaps.append(later - earlier)
            maximum_wire_gap_ns = max(local_wire_gaps, default=0)
            if capture_kernel_drops > 0:
                wire_evidence = "capture_reported_kernel_drops"
            elif len(candidate_writers) > 1:
                wire_evidence = "multiple_accepted_writers_for_topic"
            elif not writer_guid:
                wire_evidence = "insufficient_writer_topic_mapping"
            elif not times or times[0] > start_ns or times[-1] < end_ns:
                wire_evidence = "capture_does_not_bracket_probe_gap"
            elif maximum_wire_gap_ns > threshold_ns:
                wire_evidence = "wire_gap_overlaps_probe_gap"
            elif wire_inside >= 1:
                wire_evidence = "writer_sequences_observed_during_probe_gap"
            else:
                wire_evidence = "indeterminate_wire_evidence"
            result.append(
                {
                    "stream": probe_summary.stream_by_topic[topic],
                    "topic": topic,
                    "writer_guid": writer_guid,
                    "probe_gap_start_ns": start_ns,
                    "probe_gap_end_ns": end_ns,
                    "probe_gap_ns": end_ns - start_ns,
                    "probe_gap_recovered": bool(gap["recovered"]),
                    "topic_gap_threshold_ns": threshold_ns,
                    "wire_samples_inside": wire_inside,
                    "maximum_overlapping_wire_gap_ns": maximum_wire_gap_ns,
                    "wire_evidence": wire_evidence,
                }
            )
    return result


def _safe_csv_value(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_csv_value(value) for key, value in row.items()})


def write_outputs(
    output_dir: Path,
    header: PcapHeader,
    events: Sequence[RtpsEvent],
    stats: ParseStats,
    callbacks: Sequence[CallbackRow] | None,
    probe_summary: ProbeSummaryEvidence | None,
    *,
    minimum_mapping_matches: int,
    minimum_mapping_confidence: float,
    capture_kernel_drops: int,
) -> tuple[list[WriterSummary], list[WriterTopicMapping], list[dict[str, object]]]:
    if callbacks is not None and probe_summary is None:
        raise PcapError("callback mapping requires a validated probe summary")
    if callbacks is not None and stats.capped_events:
        raise PcapError("RTPS event limit was reached; refusing gap correlation")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    event_fields = list(RtpsEvent.__dataclass_fields__)
    _write_csv(
        output_dir / "rtps_events.csv",
        event_fields,
        ({field: getattr(event, field) for field in event_fields} for event in events),
    )
    summaries = summarize_writers(events)
    summary_fields = list(WriterSummary.__dataclass_fields__)
    _write_csv(
        output_dir / "rtps_writer_summary.csv",
        summary_fields,
        ({field: getattr(summary, field) for field in summary_fields} for summary in summaries),
    )
    mappings: list[WriterTopicMapping] = []
    correlations: list[dict[str, object]] = []
    if callbacks is not None:
        assert probe_summary is not None
        mappings = map_writers_to_topics(
            events,
            callbacks,
            minimum_matches=minimum_mapping_matches,
            minimum_confidence=minimum_mapping_confidence,
        )
        mapping_fields = list(WriterTopicMapping.__dataclass_fields__)
        _write_csv(
            output_dir / "writer_topic_map.csv",
            mapping_fields,
            ({field: getattr(mapping, field) for field in mapping_fields} for mapping in mappings),
        )
        correlations = correlate_probe_gaps(
            events,
            mappings,
            probe_summary,
            capture_kernel_drops=capture_kernel_drops,
        )
        correlation_fields = [
            "stream",
            "topic",
            "writer_guid",
            "probe_gap_start_ns",
            "probe_gap_end_ns",
            "probe_gap_ns",
            "probe_gap_recovered",
            "topic_gap_threshold_ns",
            "wire_samples_inside",
            "maximum_overlapping_wire_gap_ns",
            "wire_evidence",
        ]
        _write_csv(output_dir / "probe_gap_wire_evidence.csv", correlation_fields, correlations)

    report = {
        "pcap": {
            "endian": "little" if header.endian == "<" else "big",
            "timestamp_resolution_ns": header.timestamp_scale_ns,
            "linktype": header.linktype,
            "snaplen": header.snaplen,
        },
        "parse_stats": {
            **{
                field: getattr(stats, field)
                for field in ParseStats.__dataclass_fields__
                if field != "rtps_submessages"
            },
            "rtps_submessages": dict(stats.rtps_submessages),
        },
        "tcpdump_kernel_drops": capture_kernel_drops,
        "writers": [summary.__dict__ for summary in summaries],
        "writer_topic_mappings": [mapping.__dict__ for mapping in mappings],
        "probe_gap_wire_evidence": dict(Counter(str(row["wire_evidence"]) for row in correlations)),
        "probe_trace": (
            None
            if probe_summary is None
            else {
                "every_nth_callback": probe_summary.trace_every,
                "dropped_queue_rows": probe_summary.trace_dropped,
            }
        ),
    }
    with (output_dir / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summaries, mappings, correlations


def print_summary(summaries: Sequence[WriterSummary], stats: ParseStats) -> None:
    print(
        f"packets={stats.packets} udp={stats.udp_datagrams} rtps={stats.rtps_datagrams} "
        f"events={sum(stats.rtps_submessages.values())} malformed_rtps={stats.malformed_rtps}"
    )
    print(
        "writer_guid                              samples   rate_hz   p99_gap  max_gap  >75ms  missing"
    )
    for item in summaries:
        print(
            f"{item.writer_guid:<40} {item.unique_sequences:8d} {item.rate_hz:9.2f} "
            f"{item.p99_gap_ns / 1e9:8.6f} {item.maximum_gap_ns / 1e9:8.6f} "
            f"{item.gaps_over_75ms:6d} {item.forward_missing_sequences:8d}"
        )


def _pack_submessage(submessage_id: int, content: bytes, *, little: bool = True) -> bytes:
    endian = "<" if little else ">"
    flags = 0x01 if little else 0x00
    return bytes((submessage_id, flags)) + struct.pack(f"{endian}H", len(content)) + content


def _pack_sequence(sequence: int, *, little: bool) -> bytes:
    endian = "<" if little else ">"
    high, low = divmod(sequence, 1 << 32)
    return struct.pack(f"{endian}iI", high, low)


def _synthetic_data(
    sequence: int,
    *,
    prefix: bytes = bytes.fromhex("0102030405060708090a0b0c"),
    entity: bytes = bytes.fromhex("000003c2"),
    little: bool = True,
    source_ns: int | None = None,
) -> bytes:
    endian = "<" if little else ">"
    submessages = []
    if source_ns is not None:
        seconds, nanoseconds = divmod(source_ns, 1_000_000_000)
        fraction = nanoseconds * (1 << 32) // 1_000_000_000
        submessages.append(
            _pack_submessage(
                RTPS_INFO_TS,
                struct.pack(f"{endian}iI", seconds, fraction),
                little=little,
            )
        )
    content = struct.pack(f"{endian}HH", 0, 16) + b"\x00\x00\x00\x00" + entity
    content += _pack_sequence(sequence, little=little)
    submessages.append(_pack_submessage(RTPS_DATA, content, little=little))
    return b"RTPS" + b"\x02\x03" + b"\x01\x10" + prefix + b"".join(submessages)


def _synthetic_data_frag(
    sequence: int,
    fragment_start: int,
    fragments_in: int,
    *,
    fragment_size: int = 100,
    sample_size: int = 250,
    prefix: bytes = bytes.fromhex("0102030405060708090a0b0c"),
    entity: bytes = bytes.fromhex("000003c2"),
) -> bytes:
    content = struct.pack("<HH", 0, 28) + b"\x00\x00\x00\x00" + entity
    content += _pack_sequence(sequence, little=True)
    content += struct.pack("<IHHI", fragment_start, fragments_in, fragment_size, sample_size)
    return (
        b"RTPS"
        + b"\x02\x03"
        + b"\x01\x10"
        + prefix
        + _pack_submessage(RTPS_DATA_FRAG, content, little=True)
    )


def _synthetic_ipv4_udp(payload: bytes, *, ihl_words: int = 5, fragment_field: int = 0) -> bytes:
    options = b"\x00" * ((ihl_words - 5) * 4)
    udp = struct.pack(">HHHH", 7410, 7411, len(payload) + 8, 0) + payload
    total = ihl_words * 4 + len(udp)
    ip = bytearray(ihl_words * 4)
    ip[0] = (4 << 4) | ihl_words
    struct.pack_into(">H", ip, 2, total)
    struct.pack_into(">H", ip, 6, fragment_field)
    ip[8] = 64
    ip[9] = IPPROTO_UDP
    ip[12:16] = ipaddress.IPv4Address("192.168.123.10").packed
    ip[16:20] = ipaddress.IPv4Address("192.168.123.164").packed
    if options:
        ip[20:] = options
    return bytes(ip) + udp


def _synthetic_ethernet(payload: bytes, *, vlans: int = 0) -> bytes:
    header = b"\x00" * 12
    if vlans == 0:
        return header + struct.pack(">H", ETHERTYPE_IPV4) + payload
    result = header + struct.pack(">H", 0x8100)
    for index in range(vlans):
        next_type = 0x88A8 if index + 1 < vlans else ETHERTYPE_IPV4
        result += b"\x00\x01" + struct.pack(">H", next_type)
    return result + payload


def _write_synthetic_pcap(
    path: Path,
    records: Sequence[tuple[int, int, bytes, int | None]],
    *,
    endian: str = "<",
    nanoseconds: bool = True,
    linktype: int = DLT_EN10MB,
    snaplen: int = 65_535,
) -> None:
    if endian == "<":
        magic = b"\x4d\x3c\xb2\xa1" if nanoseconds else b"\xd4\xc3\xb2\xa1"
    else:
        magic = b"\xa1\xb2\x3c\x4d" if nanoseconds else b"\xa1\xb2\xc3\xd4"
    with path.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack(f"{endian}HHiiii", 2, 4, 0, 0, snaplen, linktype))
        for seconds, fraction, frame, original in records:
            original_length = len(frame) if original is None else original
            handle.write(
                struct.pack(f"{endian}IIII", seconds, fraction, len(frame), original_length)
            )
            handle.write(frame)


class _SelfTests(unittest.TestCase):
    def test_all_magic_endian_and_timestamp_resolutions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for endian in ("<", ">"):
                for nanoseconds in (False, True):
                    path = Path(directory) / f"{endian == '<'}-{nanoseconds}.pcap"
                    fraction = 123_456_789 if nanoseconds else 123_456
                    _write_synthetic_pcap(
                        path,
                        [(7, fraction, b"\x00" * 14, None)],
                        endian=endian,
                        nanoseconds=nanoseconds,
                    )
                    header, packets, handle = open_classic_pcap(path)
                    try:
                        packet = next(packets)
                    finally:
                        handle.close()
                    expected_fraction = fraction if nanoseconds else fraction * 1_000
                    self.assertEqual(packet.capture_ns, 7_000_000_000 + expected_fraction)
                    self.assertEqual(header.endian, endian)

    def test_malformed_pcap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pcap"
            path.write_bytes(PCAPNG_MAGIC + b"\x00" * 20)
            with path.open("rb") as handle, self.assertRaisesRegex(PcapError, "PCAPNG"):
                read_pcap_header(handle)
            path.write_bytes(b"\xd4\xc3\xb2\xa1")
            with path.open("rb") as handle, self.assertRaisesRegex(PcapError, "truncated"):
                read_pcap_header(handle)

    def test_link_layers_and_vlan(self) -> None:
        ip_packet = _synthetic_ipv4_udp(b"RTPS" + b"\x00" * 16)
        frames = [
            (_synthetic_ethernet(ip_packet), DLT_EN10MB),
            (_synthetic_ethernet(ip_packet, vlans=1), DLT_EN10MB),
            (_synthetic_ethernet(ip_packet, vlans=2), DLT_EN10MB),
            (b"\x00" * 14 + struct.pack(">H", ETHERTYPE_IPV4) + ip_packet, DLT_LINUX_SLL),
            (struct.pack(">H", ETHERTYPE_IPV4) + b"\x00" * 18 + ip_packet, DLT_LINUX_SLL2),
            (ip_packet, DLT_RAW),
        ]
        for frame, linktype in frames:
            udp = parse_udp_datagram(frame, linktype)
            self.assertIsNotNone(udp)
            assert udp is not None
            self.assertTrue(udp.payload.startswith(b"RTPS"))

    def test_ipv4_options_and_later_fragment(self) -> None:
        packet = _synthetic_ipv4_udp(b"RTPS" + b"\x00" * 16, ihl_words=6)
        self.assertIsNotNone(parse_udp_datagram(packet, DLT_RAW))
        later = _synthetic_ipv4_udp(b"x", fragment_field=1)
        self.assertIsNone(parse_udp_datagram(later, DLT_RAW))

    def test_rtps_data_little_and_big_endian(self) -> None:
        for little in (True, False):
            payload = _synthetic_data(0x1_0000_0002, little=little)
            udp = UdpDatagram("1.1.1.1", "2.2.2.2", 1, 2, payload, False, False)
            events = parse_rtps_events(
                CapturedPacket(0, 10, len(payload), len(payload), payload), udp
            )
            self.assertEqual(events[0].sequence, 0x1_0000_0002)
            self.assertEqual(events[0].writer_guid, "0102030405060708090a0b0c.000003c2")

    def test_data_frag_completion_and_dedupe(self) -> None:
        events: list[RtpsEvent] = []
        for index, start in enumerate((1, 2, 3)):
            payload = _synthetic_data_frag(9, start, 1)
            udp = UdpDatagram("1.1.1.1", "2.2.2.2", 1, 2, payload, False, False)
            events.extend(
                parse_rtps_events(
                    CapturedPacket(index, index * 1_000_000, len(payload), len(payload), payload),
                    udp,
                )
            )
        summary = summarize_writers(events)[0]
        self.assertEqual(summary.unique_sequences, 1)
        self.assertEqual(summary.duplicate_sequences, 0)
        self.assertEqual(summary.incomplete_fragment_samples, 0)

    def test_unilateral_gap_and_sequence_accounting(self) -> None:
        def event(prefix: str, sequence: int, timestamp: int) -> RtpsEvent:
            return RtpsEvent(
                sequence,
                20,
                timestamp,
                "1.1.1.1",
                1,
                "2.2.2.2",
                2,
                "2.3",
                "0110",
                prefix,
                "000003c2",
                f"{prefix}.000003c2",
                "DATA",
                sequence,
            )

        left = "01" * 12
        right = "02" * 12
        events = [event(left, 1, 0), event(left, 2, 1_004_000_000)]
        events += [event(right, index + 1, index * 10_000_000) for index in range(102)]
        summaries = {item.writer_guid: item for item in summarize_writers(events)}
        self.assertEqual(summaries[f"{left}.000003c2"].maximum_gap_ns, 1_004_000_000)
        self.assertEqual(summaries[f"{left}.000003c2"].gaps_over_900ms, 1)
        self.assertEqual(summaries[f"{right}.000003c2"].gaps_over_75ms, 0)

        reordered = summarize_writers(
            [event(left, sequence, index * 10_000_000) for index, sequence in enumerate((1, 3, 2))]
        )[0]
        self.assertEqual(reordered.forward_missing_sequences, 0)
        self.assertEqual(reordered.reordered_sequences, 1)

    def test_info_timestamp_mapping_and_gap_classification(self) -> None:
        prefix = bytes.fromhex("0102030405060708090a0b0c")
        events: list[RtpsEvent] = []
        base = 1_700_000_000_000_000_000
        for index in range(12):
            wire_ns = base + index * 10_000_000
            payload = _synthetic_data(index + 1, prefix=prefix, source_ns=wire_ns)
            udp = UdpDatagram("1.1.1.1", "2.2.2.2", 1, 2, payload, False, False)
            events.extend(
                parse_rtps_events(
                    CapturedPacket(index, wire_ns, len(payload), len(payload), payload), udp
                )
            )
        callbacks = [
            CallbackRow("hf_left", "rt/dex3/left/state", base, base),
            CallbackRow("hf_left", "rt/dex3/left/state", base + 110_000_000, base + 110_000_000),
        ]
        mappings = map_writers_to_topics(
            events,
            callbacks,
            tolerance_ns=2,
            minimum_matches=1,
            minimum_confidence=0.5,
        )
        self.assertEqual(mappings[0].topic, "rt/dex3/left/state")
        self.assertTrue(mappings[0].accepted)
        probe_summary = ProbeSummaryEvidence(
            trace_every=100,
            trace_dropped=0,
            stream_by_topic={"rt/dex3/left/state": "hf_left"},
            threshold_ns_by_topic={"rt/dex3/left/state": 75_000_000},
            gaps_by_topic={
                "rt/dex3/left/state": (
                    {
                        "start_wall_ns": base,
                        "end_wall_ns": base + 110_000_000,
                        "recovered": True,
                    },
                )
            },
        )
        correlations = correlate_probe_gaps(events, mappings, probe_summary)
        self.assertEqual(
            correlations[0]["wire_evidence"],
            "writer_sequences_observed_during_probe_gap",
        )
        dropped_capture = correlate_probe_gaps(
            events,
            mappings,
            probe_summary,
            capture_kernel_drops=1,
        )
        self.assertEqual(
            dropped_capture[0]["wire_evidence"],
            "capture_reported_kernel_drops",
        )

    def test_mapping_rejects_weak_and_ambiguous_source_timestamps(self) -> None:
        base = 1_700_000_000_000_000_000
        topic = "rt/dex3/left/state"
        callbacks = [CallbackRow("hf_left", topic, base, base)]
        one_writer: list[RtpsEvent] = []
        payload = _synthetic_data(1, source_ns=base)
        udp = UdpDatagram("1.1.1.1", "2.2.2.2", 1, 2, payload, False, False)
        one_writer.extend(
            parse_rtps_events(CapturedPacket(0, base, len(payload), len(payload), payload), udp)
        )
        weak = map_writers_to_topics(one_writer, callbacks, minimum_matches=5)
        self.assertFalse(weak[0].accepted)
        self.assertEqual(weak[0].evidence, "insufficient_unambiguous_timestamp_matches")

        two_writers: list[RtpsEvent] = []
        for index, prefix_byte in enumerate((1, 2)):
            candidate = _synthetic_data(
                1,
                prefix=bytes((prefix_byte,)) * 12,
                source_ns=base,
            )
            candidate_udp = UdpDatagram("1.1.1.1", "2.2.2.2", 1, 2, candidate, False, False)
            two_writers.extend(
                parse_rtps_events(
                    CapturedPacket(index, base, len(candidate), len(candidate), candidate),
                    candidate_udp,
                )
            )
        ambiguous = map_writers_to_topics(
            two_writers,
            callbacks,
            minimum_matches=1,
            minimum_confidence=0.5,
        )
        self.assertEqual(len(ambiguous), 2)
        self.assertTrue(all(not item.accepted for item in ambiguous))
        self.assertTrue(
            all(item.evidence == "ambiguous_timestamp_matches_only" for item in ambiguous)
        )

    def test_probe_summary_integrity_checks(self) -> None:
        topic = "rt/dex3/left/state"
        callbacks = [CallbackRow("hf_left", topic, 100, 90)]
        summary = {
            "schema_version": 1,
            "close_error": None,
            "trace": {
                "enabled": True,
                "every_nth_callback": 100,
                "dropped_queue_rows": 0,
            },
            "streams": {
                "hf_left": {
                    "topic": topic,
                    "gap_threshold_s": 0.075,
                    "accepted_count": 1000,
                    "gaps": [
                        {
                            "start_wall_ns": 1_000_000_000,
                            "end_wall_ns": 1_100_000_000,
                            "recovered": True,
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            evidence = read_probe_summary(path, callbacks)
            self.assertEqual(evidence.trace_every, 100)

            summary["trace"]["dropped_queue_rows"] = 1
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(PcapError, "dropped"):
                read_probe_summary(path, callbacks)

            summary["trace"]["dropped_queue_rows"] = 0
            summary["close_error"] = "RuntimeError('trace writer failed')"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(PcapError, "close error"):
                read_probe_summary(path, callbacks)

    def test_end_to_end_synthetic_pcap_outputs(self) -> None:
        base = 1_700_000_000_000_000_000
        topic = "rt/dex3/left/state"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcap = root / "synthetic.pcap"
            records = []
            callbacks = []
            for index in range(12):
                capture_ns = base + index * 10_000_000
                seconds, fraction = divmod(capture_ns, 1_000_000_000)
                payload = _synthetic_data(index + 1, source_ns=capture_ns)
                frame = _synthetic_ethernet(_synthetic_ipv4_udp(payload))
                records.append((seconds, fraction, frame, None))
                if index % 2 == 0:
                    callbacks.append(CallbackRow("hf_left", topic, capture_ns, capture_ns))
            _write_synthetic_pcap(pcap, records)
            header, events, stats = analyze_capture(pcap)
            self.assertEqual(stats.rtps_datagrams, 12)
            evidence = ProbeSummaryEvidence(
                trace_every=2,
                trace_dropped=0,
                stream_by_topic={topic: "hf_left"},
                threshold_ns_by_topic={topic: 75_000_000},
                gaps_by_topic={
                    topic: (
                        {
                            "start_wall_ns": base,
                            "end_wall_ns": base + 110_000_000,
                            "recovered": True,
                        },
                    )
                },
            )
            output = root / "analysis"
            summaries, mappings, correlations = write_outputs(
                output,
                header,
                events,
                stats,
                callbacks,
                evidence,
                minimum_mapping_matches=5,
                minimum_mapping_confidence=0.95,
                capture_kernel_drops=0,
            )
            self.assertEqual(len(summaries), 1)
            self.assertTrue(mappings[0].accepted)
            self.assertEqual(
                correlations[0]["wire_evidence"],
                "writer_sequences_observed_during_probe_gap",
            )
            for filename in (
                "analysis.json",
                "rtps_events.csv",
                "rtps_writer_summary.csv",
                "writer_topic_map.csv",
                "probe_gap_wire_evidence.csv",
            ):
                self.assertTrue((output / filename).is_file())


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap", nargs="?", type=Path, help="Classic PCAP file to analyze")
    parser.add_argument("--output-dir", type=Path, help="New directory for CSV/JSON analysis")
    parser.add_argument(
        "--callbacks-csv",
        type=Path,
        help="Sparse diagnostic callback trace used only to map RTPS writers to topics",
    )
    parser.add_argument(
        "--probe-summary-json",
        type=Path,
        help="Probe summary containing exact gap intervals and trace-integrity metadata",
    )
    parser.add_argument("--minimum-mapping-matches", type=int, default=5)
    parser.add_argument("--minimum-mapping-confidence", type=float, default=0.95)
    parser.add_argument(
        "--capture-kernel-drops",
        type=int,
        default=0,
        help="tcpdump's reported packets-dropped-by-kernel count",
    )
    parser.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help="Maximum retained RTPS events; each event consumes memory (default: 1000000)",
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic offline tests")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        if (
            args.pcap is not None
            or args.output_dir is not None
            or args.callbacks_csv is not None
            or args.probe_summary_json is not None
            or args.minimum_mapping_matches != 5
            or args.minimum_mapping_confidence != 0.95
            or args.capture_kernel_drops != 0
            or args.max_record_bytes != DEFAULT_MAX_RECORD_BYTES
            or args.max_events != DEFAULT_MAX_EVENTS
        ):
            raise SystemExit("--self-test cannot be combined with analysis arguments")
        return run_self_tests()
    if args.pcap is None:
        raise SystemExit("pcap is required unless --self-test is used")
    if args.max_record_bytes <= 0 or args.max_events <= 0:
        raise SystemExit("limits must be positive")
    if args.minimum_mapping_matches <= 0:
        raise SystemExit("--minimum-mapping-matches must be positive")
    if not 0.0 < args.minimum_mapping_confidence <= 1.0:
        raise SystemExit("--minimum-mapping-confidence must be in (0, 1]")
    if (args.callbacks_csv is None) != (args.probe_summary_json is None):
        raise SystemExit("--callbacks-csv and --probe-summary-json must be supplied together")
    if args.capture_kernel_drops < 0:
        raise SystemExit("--capture-kernel-drops must be non-negative")
    if args.max_events > DEFAULT_MAX_EVENTS:
        print(
            f"warning: retaining up to {args.max_events} RTPS events may use substantial memory",
            file=sys.stderr,
        )
    output_dir = args.output_dir or args.pcap.with_name(args.pcap.name + ".analysis")
    try:
        header, events, stats = analyze_capture(
            args.pcap,
            max_record_bytes=args.max_record_bytes,
            max_events=args.max_events,
        )
        callbacks = read_callbacks(args.callbacks_csv) if args.callbacks_csv else None
        probe_summary = (
            read_probe_summary(args.probe_summary_json, callbacks)
            if args.probe_summary_json is not None and callbacks is not None
            else None
        )
        summaries, _mappings, _correlations = write_outputs(
            output_dir,
            header,
            events,
            stats,
            callbacks,
            probe_summary,
            minimum_mapping_matches=args.minimum_mapping_matches,
            minimum_mapping_confidence=args.minimum_mapping_confidence,
            capture_kernel_drops=args.capture_kernel_drops,
        )
    except (OSError, PcapError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if stats.capped_events:
        print(
            f"warning: RTPS event limit {args.max_events} reached; output is partial",
            file=sys.stderr,
        )
    print_summary(summaries, stats)
    print(f"analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
