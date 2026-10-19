#!/usr/bin/bash
# Encode a file of base64 documents (one per line) into token ids, in parallel.
#
#   convert_enron_base64_to_ids.sh <input> <shards> <model> <prefix>
#
#   input    one base64-encoded document per line (enron_emails_to_base64.pl)
#   shards   how many pieces to split it into; each is encoded by its own
#            encode_input.pl process, at most 6 at a time
#   model    the tokeniser .model file
#   prefix   name prefix for the temporary shard files (any path)
#
# Writes <input>.encoded -- one line of comma-separated token ids per document --
# which is what dedup_singles.pl reads next. Lines come out in shard order, not
# input order; dedup_singles.pl and the trainer shuffle anyway. Run from the repo
# root: encode_input.pl loads ML/ via `use lib '.'`.
#
# The previous version could not run: it set MODELFILE from $1 (the input file),
# put ${PREFIX} inside awk's single quotes so the shards were literally named
# "${PREFIX}_0", passed $MODELFILE to a child shell that never received it, and
# its final `rm ${PREFIX}_*` matched -- and deleted -- its own output.
set -euo pipefail

usage() { echo "usage: $0 <input> <shards> <model> <prefix>" >&2; exit 1; }
[ -f "${1:-}" ]            || { echo "no file to process" >&2; usage; }
[[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "number of shards not supplied" >&2; usage; }
[ -f "${3:-}" ]            || { echo "no model file supplied" >&2; usage; }
[ -n "${4:-}" ]            || { echo "file prefix not supplied" >&2; usage; }
INPUT=$1; NSHARDS=$2; MODEL=$3; PREFIX=$4

# Leftovers from an interrupted run would be swept into the shard list below.
if compgen -G "${PREFIX}_*" > /dev/null; then
   echo "files matching ${PREFIX}_* already exist; remove them or use another prefix" >&2
   exit 1
fi

# round-robin the documents into NSHARDS files
awk -v n="$NSHARDS" -v p="$PREFIX" '{ print > (p "_" (NR-1) % n) }' "$INPUT"
shards=( "${PREFIX}"_* )

printf '%s\n' "${shards[@]}" | xargs -P 6 -I {} perl encode_input.pl "$MODEL" {}

cat "${shards[@]/%/.out}" > "${INPUT}.encoded"
rm -f "${shards[@]}" "${shards[@]/%/.out}"
echo "wrote ${INPUT}.encoded ($(wc -l < "${INPUT}.encoded") documents from ${#shards[@]} shards)"
