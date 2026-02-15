#!/usr/bin/env bash
# samply-top-symbols.sh — approximate hotspot symbols from saved Samply JSON.
#
# Works around cases where Samply save-only JSON remains unsymbolicated.
# Maps sampled leaf-frame addresses to nearest symbol using `nm -n`.
#
# Notes:
# - macOS binaries commonly require base adjustment (default 0x100000000).
# - This is a heuristic helper for quick hotspot triage, not a full symbolicator.

set -euo pipefail

# Normalize locale for tool portability/noise reduction.
export LC_ALL=C
export LANG=C

PROFILE=""
BINARY=""
TOP_N=25
THREAD_INDEX=0
BASE_ADDR="0x100000000"

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/samply-top-symbols.sh --profile <profile.json.gz|profile.json> --binary <path/to/binary> [options]

Options:
  -n, --top N          Number of rows to show (default: 25)
  --thread-index N     Thread index in Samply profile (default: 0)
  --base HEX           Base address to subtract before symbol lookup (default: 0x100000000)
  -h, --help           Show help

Example:
  ./scripts/samply-top-symbols.sh \
    --profile profiling/abc1234/lz77_encode_1MB.json.gz \
    --binary target/profiling/examples/profile \
    --top 30
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --binary)
            BINARY="$2"
            shift 2
            ;;
        -n|--top)
            TOP_N="$2"
            shift 2
            ;;
        --thread-index)
            THREAD_INDEX="$2"
            shift 2
            ;;
        --base)
            BASE_ADDR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$PROFILE" || -z "$BINARY" ]]; then
    echo "ERROR: --profile and --binary are required" >&2
    usage
    exit 1
fi

if [[ ! -f "$PROFILE" ]]; then
    echo "ERROR: profile not found: $PROFILE" >&2
    exit 1
fi
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found: $BINARY" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required" >&2
    exit 1
fi
if ! command -v nm >/dev/null 2>&1; then
    echo "ERROR: nm is required" >&2
    exit 1
fi

if [[ "$BASE_ADDR" =~ ^0[xX][0-9a-fA-F]+$ ]]; then
    BASE_DEC=$((BASE_ADDR))
else
    BASE_DEC=$BASE_ADDR
fi

if [[ "$PROFILE" == *.gz ]]; then
    PROFILE_CAT=(gzip -dc "$PROFILE")
else
    PROFILE_CAT=(cat "$PROFILE")
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
addrs_file="$tmpdir/addrs.txt"
counts_file="$tmpdir/counts.txt"
nm_file="$tmpdir/nm.txt"

"${PROFILE_CAT[@]}" | jq -r --argjson t "$THREAD_INDEX" '
  .threads[$t] as $th
  | if $th == null then empty else
      [range(0; $th.samples.length)]
      | .[]
      | ($th.samples.stack[.]) as $s
      | if $s == null then empty else $th.frameTable.address[$th.stackTable.frame[$s]] end
      | select(. != null)
    end
' > "$addrs_file"

if [[ ! -s "$addrs_file" ]]; then
    echo "No sampled addresses found (check thread index or profile file)." >&2
    exit 1
fi

sort -n "$addrs_file" | uniq -c | sort -nr | head -n "$TOP_N" > "$counts_file"

nm -n "$BINARY" | perl -ne '
  if (/^\s*([0-9A-Fa-f]+)\s+\w\s+(\S+)/) {
    print hex($1), " ", $2, "\n";
  }
' > "$nm_file"

if [[ ! -s "$nm_file" ]]; then
    echo "No symbols parsed from binary: $BINARY" >&2
    exit 1
fi

perl - "$counts_file" "$nm_file" "$BASE_DEC" <<'PERL'
use strict;
use warnings;

my ($counts_path, $nm_path, $base) = @ARGV;
$base = int($base);

open my $nm_fh, '<', $nm_path or die "open nm: $!";
my (@addr, @sym);
while (my $line = <$nm_fh>) {
    chomp $line;
    my ($a, $s) = split /\s+/, $line, 2;
    next unless defined $a && defined $s;
    push @addr, int($a);
    push @sym, $s;
}
close $nm_fh;

die "nm symbol table empty\n" unless @addr;

sub floor_idx {
    my ($x, $arr_ref) = @_;
    my @arr = @{$arr_ref};
    my ($lo, $hi) = (0, $#arr);
    my $best = -1;
    while ($lo <= $hi) {
        my $mid = int(($lo + $hi) / 2);
        if ($arr[$mid] <= $x) {
            $best = $mid;
            $lo = $mid + 1;
        } else {
            $hi = $mid - 1;
        }
    }
    return $best;
}

open my $c_fh, '<', $counts_path or die "open counts: $!";
my @rows;
while (my $line = <$c_fh>) {
    chomp $line;
    next unless $line =~ /^\s*(\d+)\s+(\d+)\s*$/;
    push @rows, [ $1, $2 ];
}
close $c_fh;

# Auto-adjust base if sampled addresses already look binary-relative.
my $max_addr = 0;
for my $r (@rows) {
    $max_addr = $r->[1] if $r->[1] > $max_addr;
}
my $min_addr = $max_addr;
for my $r (@rows) {
    $min_addr = $r->[1] if $r->[1] < $min_addr;
}

# Address-mode auto-detection:
# - profile addresses much smaller than base => profile-relative; add base
# - profile addresses around/above base      => already absolute; use as-is
# - mixed/unknown                            => assume absolute by default
my $addr_mode = "none";
if ($max_addr > 0 && $max_addr < ($base / 2)) {
    $addr_mode = "add";
} elsif ($min_addr > 0 && $min_addr < ($base / 2) && $max_addr >= ($base / 2)) {
    $addr_mode = "sub";
}

print "COUNT  PROFILE_ADDR        BINARY_ADDR         SYMBOL+OFFSET\n";
print "-----  ------------------  ------------------  ---------------------------------------------\n";
for my $r (@rows) {
    my ($count, $profile_addr) = @$r;
    my $bin_addr;
    if ($addr_mode eq "add") {
        $bin_addr = $profile_addr + $base;
    } elsif ($addr_mode eq "sub") {
        $bin_addr = $profile_addr - $base;
    } else {
        $bin_addr = $profile_addr;
    }
    if ($bin_addr < 0) {
        printf "%5d  0x%016x  %-18s  %s\n", $count, $profile_addr, "<n/a>", "<outside-base-range>";
        next;
    }

    my $idx = floor_idx($bin_addr, \@addr);
    if ($idx < 0) {
        printf "%5d  0x%016x  0x%016x  %s\n", $count, $profile_addr, $bin_addr, "<before-first-symbol>";
        next;
    }

    my $offset = $bin_addr - $addr[$idx];
    printf "%5d  0x%016x  0x%016x  %s+0x%x\n",
        $count, $profile_addr, $bin_addr, $sym[$idx], $offset;
}
PERL
