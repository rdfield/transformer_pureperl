package ML::PostFilter;
# Summary post-filter: neutralise STRUCTURED literals (emails, phones, part
# numbers / SKUs, PO/FTQ/order refs, long digit runs) that the model emitted
# but that are NOT present verbatim in the source email — i.e. fabricated or
# garbled. The model can't be trusted on rare high-entropy literals and they
# all live in metadata anyway, so a summary that OMITS one is fine; one that
# asserts a WRONG one is dangerous.
#
# Design: LIBERAL detection, STRICT membership test.
#  - detect candidate literals with a few regexes (over-matching is OK),
#  - squash both candidate and source to [a-z0-9@] (drop case/space/punct),
#  - keep the candidate iff its squashed form is a substring of the squashed
#    source (=> it was copied correctly); otherwise replace it.
# So anything the model copied right stays; only absent (invented/mangled)
# literals are touched. NAMES/ADDRESSES are deliberately NOT handled here —
# they aren't structurally detectable and need the metadata path (v2).
#
# clean_summary(summary => $text, source => $email_text, mode => 'placeholder'|'strip')
#   returns $cleaned in scalar context, ($cleaned, \@log) in list context.
#   log entries: { text => $orig, type => $t, action => 'placeholder'|'stripped' }

use strict;
use warnings;
use Exporter 'import';
our @EXPORT_OK = qw(clean_summary squash);

sub squash {
   my $s = shift;
   return '' unless defined $s;
   $s = lc $s;
   $s =~ s/[^a-z0-9\@]//g;          # keep alnum + @ (emails); drop everything else
   return $s;
}

# [ type, regex ] — tried in this order; overlaps resolved earliest-then-longest
my @PATTERNS = (
   [ 'email', qr/[A-Za-z0-9._%+\-]+\@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}/ ],
   [ 'phone', qr/\+\d[\d()\s\-]{6,}\d/ ],
   [ 'code',  qr/\b[A-Za-z0-9]+(?:[\-\x{2013}\x{2014}_+\/][A-Za-z0-9]+)+\b/ ],
   [ 'ref',   qr/\b(?:PO|POHK|FTQ|CSD|PMS|BUY|RFQ)[A-Z]*\d[\w\-]*\b/i ],
   [ 'num',   qr/\b\d{6,}\b/ ],
);

# code-shaped tokens that are really units/ranges, never literals to strip
my $UNIT = qr/^\d+[\-\x{2013}\x{2014}](?:year|years|yr|yrs|month|months|mo|day|days|week|weeks|hour|hours|h|hr|hrs)$/i;

sub clean_summary {
   my %p     = @_;
   my $sum   = defined $p{summary} ? $p{summary} : '';
   my $src   = squash($p{source} // '');
   my $mode  = $p{mode} // 'placeholder';        # 'placeholder' or 'strip'

   # 1. collect all candidate spans WITHOUT modifying the string
   my @hits;                                     # [ start, len, type, text ]
   for my $pat (@PATTERNS) {
      my ($type, $re) = @$pat;
      while ($sum =~ /$re/g) {
         my $end   = pos($sum);
         my $txt   = $&;
         my $start = $end - length($txt);
         push @hits, [ $start, length($txt), $type, $txt ];
      }
   }

   # 2. earliest-first, longest-first; drop candidates that overlap a kept one
   @hits = sort { $a->[0] <=> $b->[0] || $b->[1] <=> $a->[1] } @hits;
   my @keep;
   my $covered = -1;
   for my $h (@hits) {
      next if $h->[0] <= $covered;
      push @keep, $h;
      $covered = $h->[0] + $h->[1] - 1;
   }

   # 3. decide + apply right-to-left (so offsets stay valid)
   my @log;
   for my $h (reverse @keep) {
      my ($start, $len, $type, $txt) = @$h;
      next if $type eq 'code' && ($txt !~ /\d/ || $txt =~ $UNIT);   # unit/word, not a literal
      my $sq = squash($txt);
      next if length($sq) < 4;                    # too short to be a meaningful literal
      next if index($src, $sq) >= 0;              # present in source => copied correctly, keep

      my $repl = $mode eq 'strip' ? '' : "[$type]";
      substr($sum, $start, $len) = $repl;
      push @log, { text => $txt, type => $type,
                   action => ($mode eq 'strip' ? 'stripped' : 'placeholder') };
   }

   # 4. tidy whitespace / dangling punctuation left by removals
   if ($mode eq 'strip') {
      $sum =~ s/\s+([.,;:!?])/$1/g;               # space before punct
      $sum =~ s/\(\s*\)//g;                        # emptied parens
      $sum =~ s/\s{2,}/ /g;
      $sum =~ s/\s+([.,;:])/$1/g;
      $sum =~ s/^\s+|\s+$//g;
   } else {
      $sum =~ s/\s{2,}/ /g;
      $sum =~ s/^\s+|\s+$//g;
   }

   return wantarray ? ($sum, \@log) : $sum;
}

1;
