# Service Quota & Regional Failover Policy (Excerpt)

Provisioned primary DB read capacity is 1,200 queries per second (QPS).
Peak holiday traffic is budgeted at 3,000 QPS across checkout and catalog combined, where checkout accounts for 40% of peak traffic and catalog uses the remainder.
Policy SLA requires keeping peak utilization at or below 80% of provisioned capacity before scaling out.
Automatic cross-region failover adds an estimated 45 ms per query RTT, though replication lag under sustained write bursts above 500 writes/sec has not been benchmarked.
