# ============================================================
# KAFKA DIAGNOSTIC — Identify why consumer receives no events
# ============================================================
# Run this BEFORE the main consumer to pinpoint the issue.
# It checks: broker connection, topic list, partition offsets,
# message format, and consumer group status.
# ============================================================

import json
import time
import datetime
from kafka import KafkaConsumer, KafkaAdminClient, TopicPartition
from kafka.admin import KafkaAdminClient

from config import KAFKA_BROKERS as CFG_KAFKA_BROKERS, KAFKA_TOPIC as CFG_KAFKA_TOPIC
BROKERS = CFG_KAFKA_BROKERS or ["localhost:9092"]
EXPECTED_TOPIC = CFG_KAFKA_TOPIC


def section(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def test_broker_connection():
    """Test 1: Can we connect to each broker?"""
    section("TEST 1 — Broker connectivity")
    for broker in BROKERS:
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=[broker],
                request_timeout_ms=10000,
                api_version_auto_timeout_ms=10000,
            )
            topics = consumer.topics()
            consumer.close()
            print(f"  ✓ {broker} — connected, {len(topics)} topics visible")
        except Exception as e:
            print(f"  ✗ {broker} — FAILED: {e}")


def test_list_topics():
    """Test 2: What topics exist on the broker?"""
    section("TEST 2 — Available topics")
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=BROKERS,
            request_timeout_ms=10000,
        )
        topics = sorted(consumer.topics())
        consumer.close()

        print(f"  Found {len(topics)} topics:\n")
        for t in topics:
            marker = "  ← THIS IS YOUR TOPIC" if t == EXPECTED_TOPIC else ""
            print(f"    • {t}{marker}")

        if EXPECTED_TOPIC in topics:
            print(f"\n  ✓ Target topic '{EXPECTED_TOPIC}' EXISTS")
        else:
            print(f"\n  ✗ Target topic '{EXPECTED_TOPIC}' NOT FOUND")
            print(f"  → Possible matches:")
            for t in topics:
                if any(kw in t.lower() for kw in ["digital", "twin", "manufactured", "simulated", "sensor", "smart"]):
                    print(f"    • {t}  ← maybe this one?")
    except Exception as e:
        print(f"  ✗ Failed to list topics: {e}")


def test_topic_partitions_and_offsets():
    """Test 3: Does the topic have data? Check partition offsets."""
    section("TEST 3 — Topic partitions and offsets")
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=BROKERS,
            request_timeout_ms=10000,
        )
        topics = consumer.topics()
        if EXPECTED_TOPIC not in topics:
            print(f"  ✗ Topic '{EXPECTED_TOPIC}' not found. Skipping offset check.")
            # Try all topics that look relevant
            for t in topics:
                if any(kw in t.lower() for kw in ["digital", "twin", "manufactured", "simulated", "sensor"]):
                    _check_offsets(consumer, t)
            consumer.close()
            return

        _check_offsets(consumer, EXPECTED_TOPIC)
        consumer.close()

    except Exception as e:
        print(f"  ✗ Failed: {e}")


def _check_offsets(consumer, topic_name):
    """Check beginning and end offsets for a topic."""
    partitions = consumer.partitions_for_topic(topic_name)
    if partitions is None:
        print(f"  ✗ No partitions for topic '{topic_name}'")
        return

    print(f"\n  Topic: {topic_name}")
    print(f"  Partitions: {len(partitions)}")

    total_messages = 0
    for p in sorted(partitions):
        tp = TopicPartition(topic_name, p)
        consumer.assign([tp])

        consumer.seek_to_beginning(tp)
        begin = consumer.position(tp)

        consumer.seek_to_end(tp)
        end = consumer.position(tp)

        count = end - begin
        total_messages += count
        print(f"    Partition {p}: offset {begin}..{end} ({count} messages)")

    if total_messages == 0:
        print(f"\n  ✗ Topic '{topic_name}' is EMPTY — no messages at all")
        print(f"  → The Kafka producer might not be running")
        print(f"  → Or the topic name is wrong")
    else:
        print(f"\n  ✓ Topic '{topic_name}' has {total_messages} total messages")


def test_read_sample_messages():
    """Test 4: Try to read actual messages and show their format."""
    section("TEST 4 — Read sample messages")

    try:
        consumer = KafkaConsumer(
            bootstrap_servers=BROKERS,
            request_timeout_ms=10000,
        )
        all_topics = consumer.topics()
        consumer.close()

        # Find topics to try
        topics_to_try = []
        if EXPECTED_TOPIC in all_topics:
            topics_to_try.append(EXPECTED_TOPIC)
        # Also try similar topics
        for t in all_topics:
            if t != EXPECTED_TOPIC and any(kw in t.lower() for kw in ["digital", "twin", "manufactured", "simulated", "sensor", "smart"]):
                topics_to_try.append(t)

        if not topics_to_try:
            print(f"  ✗ No relevant topics found to read from")
            return

        for topic_name in topics_to_try[:3]:  # try up to 3 topics
            print(f"\n  Trying topic: {topic_name}")
            _read_messages(topic_name)

    except Exception as e:
        print(f"  ✗ Failed: {e}")


def _read_messages(topic_name):
    """Try to read messages from a topic."""
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=BROKERS,
            auto_offset_reset='earliest',    # read from beginning
            enable_auto_commit=False,
            consumer_timeout_ms=15000,       # wait max 15 seconds
            value_deserializer=lambda m: m,  # raw bytes
            group_id=None,                   # no group = independent read
        )

        count = 0
        for msg in consumer:
            count += 1
            ts = datetime.datetime.fromtimestamp(msg.timestamp / 1000) if msg.timestamp else "?"

            # Try to decode
            try:
                if isinstance(msg.value, bytes):
                    decoded = msg.value.decode('utf-8')
                else:
                    decoded = str(msg.value)

                # Try JSON parse
                try:
                    data = json.loads(decoded)
                    print(f"\n  Message #{count} (partition={msg.partition}, offset={msg.offset}, ts={ts}):")
                    print(f"  Format: JSON")
                    print(f"  Keys: {list(data.keys())[:15]}")

                    # Show key fields
                    if "machine_id" in data:
                        print(f"  machine_id: {data['machine_id']}")
                    if "temperature" in data:
                        print(f"  temperature: {data['temperature']}")
                    if "timestamp" in data:
                        print(f"  timestamp: {data['timestamp']}")

                    # Show full first message
                    if count == 1:
                        print(f"\n  Full message content:")
                        print(f"  {json.dumps(data, indent=4, default=str)[:1000]}")

                except json.JSONDecodeError:
                    print(f"\n  Message #{count}: NOT JSON")
                    print(f"  Raw (first 200 chars): {decoded[:200]}")

            except UnicodeDecodeError:
                print(f"\n  Message #{count}: BINARY (not UTF-8)")
                print(f"  Raw bytes (first 50): {msg.value[:50]}")

            if count >= 3:
                print(f"\n  (showing first 3 messages only)")
                break

        if count == 0:
            print(f"  ✗ No messages received within 15 seconds")
            print(f"  → Producer might be inactive")
            print(f"  → Or topic is empty")

        consumer.close()

    except Exception as e:
        print(f"  ✗ Failed to read: {e}")


def test_consumer_group_status():
    """Test 5: Check if our consumer group has committed offsets."""
    section("TEST 5 — Consumer group status")
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=BROKERS,
            request_timeout_ms=10000,
        )
        groups = admin.list_consumer_groups()
        print(f"  Active consumer groups:")
        for g in groups:
            marker = " ← OUR GROUP" if g[0] == "agentic_ai_consumer_group" else ""
            print(f"    • {g[0]} (state: {g[1]}){marker}")

        # Check our group's committed offsets
        try:
            offsets = admin.list_consumer_group_offsets("agentic_ai_consumer_group")
            if offsets:
                print(f"\n  Our group 'agentic_ai_consumer_group' committed offsets:")
                for tp, offset_meta in offsets.items():
                    print(f"    {tp.topic}[{tp.partition}] → offset {offset_meta.offset}")
            else:
                print(f"\n  Our group has no committed offsets")
                print(f"  → The consumer may have connected but never read any messages")
        except Exception:
            print(f"\n  Could not check group offsets (group may not exist yet)")

        admin.close()
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_auto_offset_reset():
    """Test 6: Try reading with auto_offset_reset='earliest' (all history)."""
    section("TEST 6 — Read with 'earliest' offset (all history)")
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=BROKERS,
            request_timeout_ms=10000,
        )
        all_topics = consumer.topics()
        consumer.close()

        # Find any topic with data
        for topic_name in sorted(all_topics):
            if any(kw in topic_name.lower() for kw in ["digital", "twin", "manufactured", "simulated", "sensor", "smart"]):
                consumer = KafkaConsumer(
                    topic_name,
                    bootstrap_servers=BROKERS,
                    auto_offset_reset='earliest',
                    enable_auto_commit=False,
                    consumer_timeout_ms=10000,
                    value_deserializer=lambda m: m,
                    group_id=None,
                )
                count = 0
                machine_ids_seen = set()
                for msg in consumer:
                    count += 1
                    try:
                        data = json.loads(msg.value.decode('utf-8') if isinstance(msg.value, bytes) else str(msg.value))
                        mid = data.get("machine_id")
                        if mid is not None:
                            machine_ids_seen.add(mid)
                    except Exception:
                        pass
                    if count >= 200:
                        break

                consumer.close()
                print(f"  Topic '{topic_name}':")
                print(f"    Messages read: {count}")
                print(f"    Unique machine IDs: {len(machine_ids_seen)}")
                if machine_ids_seen:
                    sample = sorted(machine_ids_seen)[:10]
                    print(f"    Sample IDs: {sample}")
                print()

    except Exception as e:
        print(f"  ✗ Failed: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("  KAFKA DIAGNOSTIC TOOL")
    print(f"  Brokers: {BROKERS}")
    print(f"  Expected topic: {EXPECTED_TOPIC}")
    print(f"  Time: {datetime.datetime.now()}")
    print("="*70)

    test_broker_connection()
    test_list_topics()
    test_topic_partitions_and_offsets()
    test_read_sample_messages()
    test_consumer_group_status()
    test_auto_offset_reset()

    print("\n" + "="*70)
    print("  DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\n  Kemungkinan penyebab 'Machines: 0':")
    print("  1. Topic name salah — periksa daftar topic di Test 2")
    print("  2. Topic kosong — producer tidak aktif (Test 3)")
    print("  3. auto_offset_reset='latest' tapi tidak ada data baru (Test 6)")
    print("  4. Format message bukan JSON atau field name berbeda (Test 4)")
    print("  5. Consumer group sudah commit offset di akhir (Test 5)")
    print("="*70)
