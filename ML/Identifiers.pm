package ML::Identifiers;
# What counts as an identifier, and how two of them are compared.
#
# SHARED ON PURPOSE, against the usual rule of keeping look-alike code separate:
# measure_identifiers.pl and repair_identifiers.pl are bound by a correctness
# contract. If they disagree about what an identifier is, then measuring the
# repair's effect is incoherent — the before/after numbers would be counting
# different populations. One definition is the safe option here.
#
# An identifier is a run of letters/digits/-/_/. containing AT LEAST TWO DIGITS
# and at least --min-len characters. Two digits rather than one keeps ordinary
# words with a stray digit out, while still catching "PO 10000" style references
# and SKUs like FG-700G. ISO dates and bare years are excluded by default: they
# are a different failure mode and would swamp the signal.
use strict;
use warnings;
use Exporter 'import';
our @EXPORT_OK = qw(identifiers common_prefix best_prefix_match);

sub identifiers {
   my ($text, %o) = @_;
   my $min_len = $o{min_len} // 5;
   my $dates   = $o{dates}   // 0;
   return () unless defined $text;
   my @out;
   for my $w ($text =~ m{([A-Za-z0-9][A-Za-z0-9/_.-]{3,})}g) {
      my $digits = () = $w =~ /\d/g;
      next if $digits < 2;
      unless ($dates) {
         next if $w =~ /^\d{4}-\d{2}-\d{2}$/;      # ISO date
         next if $w =~ /^(19|20)\d{2}$/;           # bare year
      }
      $w =~ s/[-_\/.]+$//;                          # trailing punctuation
      next if length($w) < $min_len;
      push @out, $w;
   }
   my %u;
   return grep { !$u{ uc $_ }++ } @out;
}

# Length of the common opening run, case-insensitive.
sub common_prefix {
   my ($a, $b) = @_;
   ($a, $b) = (uc $a, uc $b);
   my $max = length($a) < length($b) ? length($a) : length($b);
   my $i = 0;
   $i++ while $i < $max && substr($a, $i, 1) eq substr($b, $i, 1);
   return $i;
}

# Find which of @pool shares the longest opening run with $got.
#
# Returns (best, prefix_len, n_tied) where n_tied counts how many DISTINCT pool
# entries reach that same best length. n_tied > 1 means the prefix is ambiguous:
# the corpus really does contain several near-identical references (three FTQ-
# quote refs were observed, and a 5-character correct prefix cannot separate
# them), so a repair must decline rather than guess.
sub best_prefix_match {
   my ($got, $pool) = @_;
   my ($best, $blen, $tied) = (undef, 0, 0);
   for my $t (@$pool) {
      my $l = common_prefix($got, $t);
      next unless $l;
      if ($l > $blen) { ($best, $blen, $tied) = ($t, $l, 1) }
      elsif ($l == $blen && defined $best && uc($t) ne uc($best)) { $tied++ }
   }
   return ($best, $blen, $tied);
}

1;
