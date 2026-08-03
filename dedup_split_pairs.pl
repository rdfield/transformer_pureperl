#!/usr/bin/perl
# dedup_split_pairs.pl — dedup + train/val split for SLM pair data.
#
# Input: one line per pair, csv of BPE token ids: email_ids,16003,summary_ids
# Output: <train_out> and <val_out> in the same format.
#
# Dedup keys on the EMAIL side only — the same email with a different
# summary is still train/val leakage. Two passes:
#   1. exact dedup: hash of the full email id sequence
#   2. near dedup: MinHash (16 hashes, 4 bands) over 4-gram shingles of
#      the email ids; candidate pairs sharing a band bucket get a true
#      Jaccard check against --jaccard (default 0.8). First occurrence
#      wins; later near-duplicates are dropped.
# Then a seeded shuffle and split (--val-frac, default 0.1).
#
# usage: perl dedup_split_pairs.pl <pairs.csv> <train_out> <val_out>
#           [--val-frac 0.1] [--seed 42] [--jaccard 0.8]
use Modern::Perl;
use lib '.';

my $SEP_ID = 16003;

my ($infile, $train_out, $val_out, @rest) = @ARGV;
die "usage: perl dedup_split_pairs.pl <pairs.csv> <train_out> <val_out> [--val-frac F] [--seed N] [--jaccard F]\n"
   unless defined($infile) and -e $infile and defined($train_out) and defined($val_out);

my %opt = ('--val-frac' => 0.1, '--seed' => 42, '--jaccard' => 0.5);
while (@rest) {
   my $k = shift @rest;
   die "unknown or valueless option $k\n" unless exists $opt{$k} and @rest;
   $opt{$k} = shift @rest;
}

# ---- load ------------------------------------------------------------------
open(my $in, '<', $infile) or die "cannot open $infile: $!\n";
my @lines;      # raw lines, kept verbatim for output
my @emails;     # arrayref of email ids per line
my $malformed = 0;
while (my $line = <$in>) {
   chomp $line;
   next unless length $line;
   my @ids = split /,/, $line;
   my ($sep_at) = grep { $ids[$_] == $SEP_ID } 0 .. $#ids;
   if (!defined($sep_at) or $sep_at == 0 or $sep_at == $#ids) {
      $malformed++;
      next;
   }
   push @lines, $line;
   push @emails, [ @ids[0 .. $sep_at - 1] ];
}
close $in;
say "loaded: " . scalar(@lines) . " pairs ($malformed malformed skipped)";

# ---- pass 1: exact dedup ---------------------------------------------------
my %seen_exact;
my @keep_idx;
my $exact_dropped = 0;
foreach my $i (0 .. $#lines) {
   my $key = join(',', @{$emails[$i]});
   if ($seen_exact{$key}++) {
      $exact_dropped++;
   } else {
      push @keep_idx, $i;
   }
}
say "exact dedup: dropped $exact_dropped";

# ---- pass 2: near dedup (MinHash bands + Jaccard confirm) ------------------
my $NUM_HASH = 16;
my $BANDS    = 8;            # 2 rows per band: surfaces J=0.5 candidates ~90% of the time
my @hash_salt = map { 0x9e3779b9 * ($_ + 1) & 0xffffffff } 0 .. $NUM_HASH - 1;

sub shingles {
   my ($ids) = @_;
   my %sh;
   if (@$ids < 4) {
      $sh{join(',', @$ids)} = 1;
   } else {
      $sh{join(',', @{$ids}[$_ .. $_ + 3])} = 1 for 0 .. $#$ids - 3;
   }
   return \%sh;
}

sub minhash_sig {
   my ($sh) = @_;
   my @sig = (0xffffffff) x $NUM_HASH;
   foreach my $s (keys %$sh) {
      my $h = unpack('%32N*', $s) ^ 0;   # cheap base hash of the shingle text
      $h = ($h * 2654435761) & 0xffffffff;
      foreach my $k (0 .. $NUM_HASH - 1) {
         my $hk = ($h ^ $hash_salt[$k]) & 0xffffffff;
         $hk = ($hk * 40503 + 0x5bd1) & 0xffffffff;
         $sig[$k] = $hk if $hk < $sig[$k];
      }
   }
   return \@sig;
}

sub jaccard {
   my ($a, $b) = @_;
   my $inter = 0;
   $inter += exists $b->{$_} for keys %$a;
   my $union = scalar(keys %$a) + scalar(keys %$b) - $inter;
   return $union ? $inter / $union : 1;
}

my %band_buckets;   # band key -> [kept indices]
my @final_idx;
my $near_dropped = 0;
my %shcache;
foreach my $i (@keep_idx) {
   my $sh  = shingles($emails[$i]);
   my $sig = minhash_sig($sh);
   my @keys;
   foreach my $b (0 .. $BANDS - 1) {
      my $rows = $NUM_HASH / $BANDS;
      push @keys, "$b:" . join('_', @{$sig}[$b * $rows .. ($b + 1) * $rows - 1]);
   }
   my $dup = 0;
   my %cand;
   foreach my $k (@keys) {
      $cand{$_} = 1 for @{$band_buckets{$k} // []};
   }
   foreach my $j (keys %cand) {
      if (jaccard($sh, $shcache{$j}) >= $opt{'--jaccard'}) {
         $dup = 1;
         last;
      }
   }
   if ($dup) {
      $near_dropped++;
   } else {
      push @final_idx, $i;
      $shcache{$i} = $sh;
      push @{$band_buckets{$_}}, $i for @keys;
   }
}
say "near dedup:  dropped $near_dropped (jaccard >= $opt{'--jaccard'} on email shingles)";
say "kept:        " . scalar(@final_idx);

# ---- seeded shuffle + split ------------------------------------------------
srand($opt{'--seed'});
for (my $i = $#final_idx; $i > 0; $i--) {
   my $j = int(rand($i + 1));
   @final_idx[$i, $j] = @final_idx[$j, $i];
}
my $n_val = int(scalar(@final_idx) * $opt{'--val-frac'});
my @val_idx   = @final_idx[0 .. $n_val - 1];
my @train_idx = @final_idx[$n_val .. $#final_idx];

open(my $tr, '>', $train_out) or die "cannot write $train_out: $!\n";
print $tr "$lines[$_]\n" for @train_idx;
close $tr;
open(my $va, '>', $val_out) or die "cannot write $val_out: $!\n";
print $va "$lines[$_]\n" for @val_idx;
close $va;
say "split:       " . scalar(@train_idx) . " train -> $train_out, " . scalar(@val_idx) . " val -> $val_out (seed $opt{'--seed'})";
