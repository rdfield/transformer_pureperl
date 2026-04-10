#!/usr/bin/perl
# test_tokeniser.pl — correctness tests for ML::Tokeniser
#
# Assumed ML::Tokeniser interface:
#
#   $tok = ML::Tokeniser->new(vocab_size => N)
#          Target vocabulary size N > 256.
#          Base vocab is 256 UTF-8 byte values (IDs 0-255).
#          N-256 merges will be performed during train().
#
#   $tok->train(text => $text)
#          BPE training on a UTF-8 string.
#          Merges stored in training order; index 0 = highest priority.
#
#   $ids = $tok->encode(text => $text)
#          Returns arrayref of integer token IDs.
#          Applies merges in training order (lowest index first).
#
#   $text = $tok->decode(ids => \@ids)
#          Returns UTF-8 string.
#
#   $tok->save(file => $path)
#          Writes vocab + merges to a JSON file.
#
#   $tok->load(file => $path)
#          Replaces current state from a JSON file.
#
#   $n = $tok->vocab_size()
#          Returns 256 + number of merges trained so far.
#
#   $merges = $tok->merges()
#          Returns arrayref of merge records in training order.
#          Each record: { pair => [$id1, $id2], idx => $new_id }
#
# Tests:
#   1.  Smoke — train + vocab_size check
#   2.  Round-trip on training text
#   3.  Round-trip on unseen text
#   4.  Known first merge (controlled corpus)
#   5.  Merge applied correctly by encode
#   6.  Merge count matches vocab_size target
#   7.  All token IDs in valid range after encode
#   8.  Merge priority order: lower-index merge applied before higher-index
#   9.  UTF-8 multi-byte round-trip
#   10. Save / load round-trip (same encode output)
#   11. Edge cases: empty string, single ASCII byte, single non-ASCII char

use Modern::Perl;
use utf8;
use lib '.';
use ML::Tokeniser;
use File::Temp qw(tempfile);

binmode(STDOUT, ':utf8');

my ($PASS, $FAIL) = (0, 0);

# ── helpers ───────────────────────────────────────────────────────────────────

sub pass { printf "  PASS  %s\n", $_[0]; $PASS++ }
sub fail { printf "  FAIL  %s\n", $_[0]; $FAIL++ }

sub ok {
    my ($label, $cond) = @_;
    $cond ? pass($label) : fail($label);
}

sub ok_eq {
    my ($label, $got, $expected) = @_;
    if ($got eq $expected) {
        pass($label);
    } else {
        fail("$label\n         got:      " . substr($got, 0, 80) .
             "\n         expected: " . substr($expected, 0, 80));
    }
}

sub ok_ids_eq {
    my ($label, $got, $expected) = @_;
    my $gs = join(",", @$got);
    my $es = join(",", @$expected);
    if ($gs eq $es) {
        pass($label);
    } else {
        fail("$label\n         got:      $gs\n         expected: $es");
    }
}

# ── shared corpus ─────────────────────────────────────────────────────────────

# Nathan Reed quote — same text used in basic_tokeniser.pl
my $TRAIN_TEXT = q{Unicode! The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to support Unicode in our software. But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception. But today I'm going to take a deep breath and try to explain the absolute minimum that a programmer needs to know about Unicode. Unicode is a standard for encoding, transmitting and processing text. The most important thing to understand about Unicode is that it is fundamentally a mapping of integers to characters.};

my $UNSEEN_TEXT = q{Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and distinguished from characters of other scripts by their properties.};

# ─────────────────────────────────────────────────────────────────────────────
say "=" x 70;
say "TEST 1: Smoke — train and vocab_size";
say "=" x 70;

{
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);
    ok("vocab_size == 276", $tok->vocab_size() == 276);
    ok("merges() has 20 entries", scalar(@{$tok->merges()}) == 20);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 2: Round-trip on training text";
say "=" x 70;

{
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);
    my $ids      = $tok->encode(text => $TRAIN_TEXT);
    my $decoded  = $tok->decode(ids  => $ids);
    ok_eq("decode(encode(train_text)) eq train_text", $decoded, $TRAIN_TEXT);
    ok("encoded length < byte length (compression happened)",
        scalar(@$ids) < length(Encode::encode_utf8($TRAIN_TEXT)));
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 3: Round-trip on unseen text";
say "=" x 70;

{
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);
    my $ids     = $tok->encode(text => $UNSEEN_TEXT);
    my $decoded = $tok->decode(ids  => $ids);
    ok_eq("decode(encode(unseen_text)) eq unseen_text", $decoded, $UNSEEN_TEXT);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 4: Known first merge on controlled corpus";
say "=" x 70;

# "ababababab" → bytes [97,98] × 5.  Most frequent pair: (97,98) × 5.
# With vocab_size=257 (one merge), token 256 must represent bytes [97,98].
{
    my $corpus = "ababababab";
    my $tok = ML::Tokeniser->new(vocab_size => 257);
    $tok->train(text => $corpus);

    my $merges = $tok->merges();
    ok("exactly one merge trained", scalar(@$merges) == 1);
    my $m = $merges->[0];
    ok("first merge pair is (97, 98) i.e. 'a'+'b'",
        $m->{pair}[0] == 97 && $m->{pair}[1] == 98);
    ok("first merge new ID is 256", $m->{value} == 256);

    # decode token 256 should give "ab"
    my $decoded = $tok->decode(ids => [256]);
    ok_eq("decode([256]) eq 'ab'", $decoded, "ab");
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 5: encode applies merge correctly";
say "=" x 70;

{
    my $corpus = "ababababab";
    my $tok257 = ML::Tokeniser->new(vocab_size => 257);  # one merge: (97,98)→256
    $tok257->train(text => $corpus);

    # "ab" should compress to a single token 256
    { my $r = $tok257->encode(text => "ab");
      ok_ids_eq("encode('ab') == [256]", $r, [256]); }

    # "ababab" should become [256,256,256]
    { my $r = $tok257->encode(text => "ababab");
      ok_ids_eq("encode('ababab') == [256,256,256]", $r, [256, 256, 256]); }

    # "a" is not part of a merged pair; stays as byte 97
    { my $r = $tok257->encode(text => "a");
      ok_ids_eq("encode('a') == [97]", $r, [97]); }

    # "ba" — reversed pair — must NOT be merged; stays [98, 97]
    { my $r = $tok257->encode(text => "ba");
      ok_ids_eq("encode('ba') == [98,97]  (pair not reversed)", $r, [98, 97]); }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 6: Merge count matches vocab_size target";
say "=" x 70;

{
    for my $vs (257, 260, 270, 276) {
        my $tok = ML::Tokeniser->new(vocab_size => $vs);
        $tok->train(text => $TRAIN_TEXT);
        my $n = scalar(@{$tok->merges()});
        ok("vocab_size=$vs → merges=${\($vs-256)} (got $n)",
            $tok->vocab_size() == $vs && $n == $vs - 256);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 7: All token IDs in valid range [0, vocab_size-1]";
say "=" x 70;

{
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);
    my $vs = $tok->vocab_size();

    for my $text ($TRAIN_TEXT, $UNSEEN_TEXT,
                  "Hello, world!", "0123456789") {
        my $ids = $tok->encode(text => $text);
        my @bad = grep { $_ < 0 || $_ >= $vs } @$ids;
        ok("no out-of-range IDs for: " . substr($text, 0, 30) . "...",
            scalar(@bad) == 0);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 8: Merge chaining — successive merges applied in training order";
say "=" x 70;

# Training on "abcabcabc":
#   Round 1 pairs: (97,98)=3, (98,99)=3, (99,97)=2.  Tie (97,98) vs (98,99).
#   mergeidx tiebreak: (97,98) encountered at position 0 first → merge 0: (97,98)→256
#   After merge: [256,99,256,99,256,99]
#   Round 2: only pair is (256,99)=3 → merge 1: (256,99)→257
#
# So: 256 = "ab",  257 = "abc" (= 256 + "c")
#
# encode("abc") must chain both merges:
#   [97,98,99] → apply (97,98)→256 → [256,99] → apply (256,99)→257 → [257]
# encode("bc") has no applicable merge → stays [98,99]
# decode([257]) must recover "abc"
{
    my $corpus = "abcabcabc";
    my $tok = ML::Tokeniser->new(vocab_size => 258);  # 2 merges
    $tok->train(text => $corpus);

    my $merges = $tok->merges();
    ok("two merges trained", scalar(@$merges) == 2);

    # First merge should be (97,98)→256 (wins tie by first-occurrence mergeidx)
    my $m0 = $merges->[0];
    ok("merge 0 pair is (97,98) i.e. 'a'+'b'",
        $m0->{pair}[0] == 97 && $m0->{pair}[1] == 98 && $m0->{value} == 256);

    # Second merge should be (256,99)→257
    my $m1 = $merges->[1];
    ok("merge 1 pair is (256,99) i.e. token256+'c'",
        $m1->{pair}[0] == 256 && $m1->{pair}[1] == 99 && $m1->{value} == 257);

    # encode("abc"): both merges chain → single token 257
    { my $r = $tok->encode(text => "abc");
      ok_ids_eq("encode('abc') chains both merges → [257]", $r, [257]); }

    # encode("bc"): neither merge applies → raw bytes
    { my $r = $tok->encode(text => "bc");
      ok_ids_eq("encode('bc') has no merge → [98,99]", $r, [98, 99]); }

    # decode([257]) must recover "abc"
    ok_eq("decode([257]) eq 'abc'", $tok->decode(ids => [257]), "abc");
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 9: UTF-8 multi-byte round-trip";
say "=" x 70;

{
    my $unicode_text = "Unicode\x{00e9}l\x{00e8}ve caf\x{00e9} na\x{00ef}ve";  # accented Latin
    my $emoji_text   = "\x{1f600}\x{1f601}\x{1f602}";  # emoji
    my $cjk_text     = "\x{4e2d}\x{6587}\x{6d4b}\x{8bd5}";  # Chinese

    # Train on plain ASCII; the module must still round-trip multi-byte chars
    # via the base 256-byte vocabulary (no merges needed for correctness).
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);

    for my $text ($unicode_text, $emoji_text, $cjk_text) {
        my $ids     = $tok->encode(text => $text);
        my $decoded = $tok->decode(ids  => $ids);
        ok_eq("round-trip: " . $text, $decoded, $text);
    }
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 10: Save / load round-trip";
say "=" x 70;

{
    # save() appends .model (and .vocab); load() requires the .model path
    my ($fh, $tmpfile) = tempfile(UNLINK => 0);
    close $fh;
    unlink $tmpfile;  # save() creates $tmpfile.model itself

    my $tok_a = ML::Tokeniser->new(vocab_size => 276);
    $tok_a->train(text => $TRAIN_TEXT);
    $tok_a->save(file => $tmpfile);          # writes $tmpfile.model + $tmpfile.vocab

    my $tok_b = ML::Tokeniser->new(vocab_size => 276);
    $tok_b->load(file => "$tmpfile.model");  # load() requires the .model suffix

    ok("loaded vocab_size matches", $tok_b->vocab_size() == 276);
    ok("loaded merge count matches", scalar(@{$tok_b->merges()}) == 20);

    # Same encode output from both instances
    for my $text ($TRAIN_TEXT, $UNSEEN_TEXT, "hello world") {
        my $ids_a = join(",", @{$tok_a->encode(text => $text)});
        my $ids_b = join(",", @{$tok_b->encode(text => $text)});
        ok("encode after save/load matches for: " . substr($text, 0, 30) . "...",
            $ids_a eq $ids_b);
    }

    # decode from loaded instance still round-trips
    my $ids     = $tok_b->encode(text => $UNSEEN_TEXT);
    my $decoded = $tok_b->decode(ids  => $ids);
    ok_eq("decode after load round-trips unseen text", $decoded, $UNSEEN_TEXT);

    unlink "$tmpfile.model", "$tmpfile.vocab";
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
say "TEST 11: Edge cases";
say "=" x 70;

{
    my $tok = ML::Tokeniser->new(vocab_size => 276);
    $tok->train(text => $TRAIN_TEXT);

    # Empty string
    my $empty = $tok->encode(text => "");
    ok("encode('') returns empty list", scalar(@$empty) == 0);
    ok_eq("decode([]) returns ''", $tok->decode(ids => []), "");

    # Single ASCII byte — must be its own byte value
    my $ids_a = $tok->encode(text => "a");
    ok("encode('a') is [97]", scalar(@$ids_a) == 1 && $ids_a->[0] == 97);

    # Single non-ASCII character (é = U+00E9 = bytes 0xC3 0xA9 = 195, 169)
    my $e_acute = "\x{00e9}";
    my $ids_e   = $tok->encode(text => $e_acute);
    my $decoded = $tok->decode(ids  => $ids_e);
    ok_eq("round-trip single non-ASCII char é", $decoded, $e_acute);

    # All 256 byte values individually decodable
    my $all_bytes_ok = 1;
    for my $b (0 .. 255) {
        my $d = $tok->decode(ids => [$b]);
        $all_bytes_ok = 0 unless defined $d;
    }
    ok("all 256 base byte IDs decode without error", $all_bytes_ok);
}

# ─────────────────────────────────────────────────────────────────────────────
say "\n" . "=" x 70;
printf "TOTAL: %d passed, %d failed\n", $PASS, $FAIL;
say "OVERALL: " . ($FAIL == 0 ? "PASS" : "FAIL");
say "=" x 70;
