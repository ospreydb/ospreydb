# OspreyDB Pro — End-User License Agreement (EULA)

**Effective Date:** The date of purchase or first download.

**Licensor:** OspreyDB Inc.

## 1. Grant of License

Subject to payment of the applicable subscription fee and compliance with this EULA, Licensor grants
Licensee a **non-exclusive, non-transferable, worldwide annual subscription license** to install and
run one (1) instance of the OspreyDB Pro binary per licensed node, solely for Licensee's internal
business operations.

- **Community / Personal Free Tier:** Running `osprey-pro` without a license key activates a
  `Single-Node Community Tier` with a hard technical cap of **5,000,000 (5M) vectors** and
  local/single-writer storage. The free tier includes the full WAL, LSM Delta Buffer, HTTP API
  (`:6333`), and Auto-Embeddings `/v1/embeddings` at no cost.
- **Founding Tier:** Early Access discounted license. Unlocks unlimited vectors (>5M),
  multi-node S3 synchronization via `manifest.json`, and the Apache Arrow Flight server.
- **Enterprise Tier:** Full commercial license. Unlocks unlimited vectors, multi-node S3 CAS
  with optimistic locking (`ETag` / `If-Match`), Apache Arrow Flight, and optional 24/7 SLA.

## 2. 100% BYOC (Bring Your Own Cloud)

OspreyDB Pro is deployed entirely inside Licensee's own cloud accounts (AWS, GCP, Azure) or
on-premises infrastructure. Licensor has **no access** to Licensee's S3 buckets, compute
instances, IAM roles, or VPC networks. The offline Ed25519 license verification performs
**zero phone-home calls** — there is no telemetry, no license check server, and no remote
kill-switch.

## 3. Annual Subscription with 30-Day Renewal Grace Period

Each commercial license is an **annual subscription** that includes **all software updates**
(major, minor, and patch releases) throughout the active 12-month term. To prevent accidental
production downtime if payment renewal is delayed, the offline Ed25519 verifier enforces a
**30-day renewal grace period** past `expires_at` (emitting explicit warning logs). Once the
30-day grace period elapses without a renewed key, Pro-tier startup is blocked until a valid
subscription key is activated, while existing on-disk/S3 data remains readable via the open-core
`osprey` CLI.

## 4. Restrictions

Licensee shall not:
- Reverse-engineer, decompile, or disassemble the Software except as expressly permitted by law.
- Remove, obscure, or alter any proprietary notices or labels.
- Sublicense, rent, lease, or resell the Software as a stand-alone service.
- Use the Software to build a competing vector database SaaS offering.

## 5. No Warranty

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS IS" WITHOUT
WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

## 6. Limitation of Liability

IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, INCLUDING LOST PROFITS OR DATA, ARISING OUT OF THE USE OR INABILITY TO USE THE SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. LICENSOR'S TOTAL LIABILITY SHALL NOT EXCEED
THE LICENSE FEES PAID BY LICENSEE IN THE 12 MONTHS PRECEDING THE CLAIM.

## 7. Support

Support is provided as described in [SUPPORT.md](./SUPPORT.md). All support is strictly
asynchronous (GitHub Issues / email, < 48 business hours). 24/7 on-call SLA is available as a
separately priced add-on for Enterprise license holders.

## 8. Termination

This EULA terminates automatically if Licensee breaches any material term and fails to cure
within 30 days of written notice. Upon termination, Licensee must cease all use and destroy all
copies. Sections 4-8 survive termination.

## 9. Governing Law

This EULA is governed by the laws of the State of Delaware, USA, without regard to conflict of
law principles.

---

**OspreyDB Inc.**
`legal@ospreydb.dev`
[https://ospreydb.dev](https://ospreydb.dev)