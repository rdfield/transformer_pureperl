#!/usr/bin/perl
# Bridge MVEncDecInfer output into the CSV formats the identifier tooling reads.
#
# MVEncDecInfer prints, per example:
#     === example N (thread T tok, gen G tok, logprob L) ===
#     GEN <comma-separated ids>
#     REF <comma-separated ids>
# and processes the FIRST --n lines of the pairs file IN ORDER, so generations are
# line-aligned with the input. The SOURCE THREAD is the part of each pairs line
# before the [SEP], which is what makes the repair pass possible at all — the
# repair matches emitted identifiers against identifiers in the source, and
# nothing else in the pipeline carries the source text alongside a generation.
#
# ALIGNMENT IS VERIFIED, NOT ASSUMED. MVEncDecInfer skips any line with no [SEP]
# or with [SEP] at index 0, so naive line-counting can silently pair every
# generation with the wrong thread. This replicates that skip rule AND checks the
# REF ids it printed against the reference decoded from the pairs line. A mismatch
# aborts: misalignment would make every downstream number meaningless while
# looking entirely plausible.
# (Measured 2026-10-05: train_16000_finetune.txt and val_16000_finetune.txt have
# zero skippable lines, so the naive assumption happens to hold on this data.
# audit_sample.pl relies on it. The check is for the day it stops holding.)
#
# Writes:
#   <out>_pairs.csv  id,band,source_b64,reference_b64   (measure/repair read SOURCE here)
#   <out>_gen.csv    id,model,generation_b64,0,0,0,stop,n/a
#
# Then:
#   perl measure_identifiers.pl  OUT_pairs.csv OUT_gen.csv              # before
#   perl repair_identifiers.pl   OUT_pairs.csv OUT_gen.csv OUT_rep.csv  # repair
#   perl measure_identifiers.pl  OUT_pairs.csv OUT_gen.csv OUT_rep.csv  # before vs after
#
#   perl gen_to_bakeoff.pl --pairs val.txt --gen gen.txt --out /tmp/ft
#        [--vocab model] [--label NAME]
#
use Modern::Perl;
use Getopt::Long;
use lib '/home/mvine/embedding';
use Text::CSV;
use MIME::Base64 qw(encode_base64);
use ML::Tokeniser;

my ($pairs, $gen_file, $out) = ('', '', '');
my $VOCAB = '/home/mvine/shared/rdf/pretrain_data/cosmo_client_blended2.model';
my $LABEL = 'student';
my $SEP   = 16003;
GetOptions('pairs=s' => \$pairs, 'gen=s' => \$gen_file, 'out=s' => \$out,
           'vocab=s' => \$VOCAB, 'label=s' => \$LABEL, 'sep=i' => \$SEP)
   or die usage();
die usage() unless $pairs && $gen_file && $out;
die "$pairs not found\n"    unless -e $pairs;
die "$gen_file not found\n" unless -e $gen_file;

my $t = ML::Tokeniser->new(vocab_size => 16000);
$t->load(file => $VOCAB);

# ---- generations, in file order ----
my (@gen, @ref);
open my $gf, '<', $gen_file or die "$gen_file: $!";
while (my $l = <$gf>) {
   push @gen, $1 if $l =~ /^GEN\s*(.*)$/;
   push @ref, $1 if $l =~ /^REF\s*(.*)$/;
}
close $gf;
die "no GEN lines in $gen_file\n" unless @gen;
die "GEN/REF count mismatch (" . scalar(@gen) . "/" . scalar(@ref) . ")\n"
   unless @gen == @ref;
say "generations: " . scalar(@gen);

# ---- pairs, applying MVEncDecInfer's own skip rule ----
my (@src_ids, @ref_ids, @skipped);
open my $pf, '<', $pairs or die "$pairs: $!";
my $lineno = 0;
while (my $l = <$pf>) {
   $lineno++;
   last if @src_ids >= @gen;
   chomp $l;
   next unless length $l;
   my @ids = grep { /^-?\d+$/ } split /,/, $l;
   my ($sep) = grep { $ids[$_] == $SEP } 0 .. $#ids;
   if (!defined $sep || $sep == 0) { push @skipped, $lineno; next }
   push @src_ids, [ @ids[ 0 .. $sep - 1 ] ];
   push @ref_ids, [ @ids[ $sep + 1 .. $#ids ] ];
}
close $pf;
say "pairs read: " . scalar(@src_ids) . (@skipped ? "  (skipped lines: @skipped)" : "");
die "ran out of pairs: " . scalar(@src_ids) . " for " . scalar(@gen) . " generations\n"
   if @src_ids < @gen;

# ---- verify alignment against the REF the generator printed ----
my $bad = 0;
for my $i (0 .. $#gen) {
   my $printed = join ',', grep { /^-?\d+$/ } split /,/, ($ref[$i] // '');
   my $fromfile = join ',', @{ $ref_ids[$i] };
   next if $printed eq $fromfile;
   $bad++;
   say "ALIGNMENT MISMATCH at example " . ($i + 1) . ":" if $bad <= 3;
   say "  generator REF: " . substr($printed, 0, 70)   if $bad <= 3;
   say "  pairs file   : " . substr($fromfile, 0, 70)  if $bad <= 3;
}
die "\n$bad of " . scalar(@gen) . " examples are MISALIGNED — refusing to write.\n" .
    "Every downstream identifier number would be measured against the wrong thread.\n"
   if $bad;
say "alignment verified: all " . scalar(@gen) . " REFs match the pairs file";

# ---- write ----
my $pcsv = Text::CSV->new({ binary => 1, eol => "\n" });
my $gcsv = Text::CSV->new({ binary => 1, eol => "\n" });
open my $po, '>', "${out}_pairs.csv" or die $!;
open my $go, '>', "${out}_gen.csv"   or die $!;

my $dec = sub {
   my @i = grep { /^\d+$/ } @{ $_[0] };
   return '' unless @i;
   my $s = $t->decode(ids => \@i);
   utf8::encode($s) if utf8::is_utf8($s);
   return $s;
};
my $b64 = sub { return encode_base64($_[0], '') };

for my $i (0 .. $#gen) {
   my $src = $dec->($src_ids[$i]);
   my $rf  = $dec->($ref_ids[$i]);
   my @g   = grep { /^\d+$/ } split /,/, ($gen[$i] // '');
   my $gn  = $dec->(\@g);
   my $id  = $i + 1;
   # band = source length, so measure_identifiers' report is readable
   $pcsv->print($po, [ $id, scalar(@{ $src_ids[$i] }) . 'tok', $b64->($src), $b64->($rf) ]);
   $gcsv->print($go, [ $id, $LABEL, $b64->($gn), 0, 0, 0, 'stop', 'n/a' ]);
}
close $po; close $go;

say "wrote ${out}_pairs.csv  (source + reference)";
say "wrote ${out}_gen.csv    (generations, label '$LABEL')";

sub usage {
   return "usage: $0 --pairs val.txt --gen gen.txt --out PREFIX " .
          "[--vocab model] [--label NAME] [--sep 16003]\n";
}
