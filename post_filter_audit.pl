#!/usr/bin/perl
# Post-filter audit harness. Sits where decode_gen.pl did in the pipe, but
# ALSO strips fabricated/mangled structured literals from each generated
# summary (via ML::PostFilter) using the SOURCE EMAIL as ground truth.
#
#   cd embedding_cpp
#   ./build/MVSLMInfer --model slm_fsl_pw05_j07.safetensors \
#       --input /home/mvine/embedding/slm_val_j07.csv --n 100 --rep_penalty 1.2 \
#     | perl /home/mvine/embedding/post_filter_audit.pl \
#         --val /home/mvine/embedding/slm_val_j07.csv > filtered_100.txt
#   perl /home/mvine/embedding/audit_tally.pl filtered_100.txt
#
# The source email for example N is line N of --val: the ids BEFORE the first
# SEP (16003) are the email the model actually saw. GEN is the cleaned summary.
# A per-example note + a run summary go to STDERR (stdout stays audit-clean).

use strict;
use warnings;
use feature 'say';
use lib '/home/mvine/embedding';
use ML::Tokeniser;
use ML::PostFilter qw(clean_summary);

binmode(STDOUT, ':utf8');
binmode(STDERR, ':utf8');

my $SEP = 16003;
my $val_path = '/home/mvine/embedding/slm_val_j07.csv';
my $vocab_path;

my $mode     = 'placeholder';
for (my $i = 0; $i < @ARGV; $i++) {
   if    ($ARGV[$i] eq '--val')  { $val_path = $ARGV[++$i]; }
   elsif ($ARGV[$i] eq '--mode') { $mode     = $ARGV[++$i]; }
   elsif ($ARGV[$i] eq '--vocab') { $vocab_path     = $ARGV[++$i]; }
   else { die "unknown arg $ARGV[$i]\n"; }
}
die "mode must be placeholder|strip\n" unless $mode =~ /^(placeholder|strip)$/;
die "--vocab must be a minbpe file" unless defined($vocab_path) and -e $vocab_path;

my $t = ML::Tokeniser->new(vocab_size => 276);
$t->load(file => $vocab_path);

# source email text per val line (decode the ids before the first SEP)
open my $vf, '<', $val_path or die "open $val_path: $!\n";
my @src_email;
while (my $line = <$vf>) {
   chomp $line;
   my @ids = grep { length && /^\d+$/ } split /,/, $line;
   my @email;
   for my $id (@ids) { last if $id == $SEP; push @email, $id; }
   push @src_email, (@email ? $t->decode(ids => \@email) : '');
}
close $vf;

sub ids_to_text {
   my @i = grep { length && /^\d+$/ && $_ < 16000 } split /,/, ($_[0] // '');
   my $txt = @i ? $t->decode(ids => \@i) : '';
   $txt =~ s/\s+/ /g;
   $txt =~ s/^\s+|\s+$//g;
   return $txt;
}

my ($cur_n, $touched, $total_lits, %by_type, $examples) = (0, 0, 0);
while (my $line = <STDIN>) {
   chomp $line;
   if ($line =~ /^=== example\s+(\d+)\b/) {
      $cur_n = $1; $examples++;
      say $line;
   }
   elsif ($line =~ /^GEN ?(.*)$/) {
      my $summary = ids_to_text($1);
      my $source  = $src_email[$cur_n - 1];
      if (!defined $source) {
         warn "  ! example $cur_n: no source line $cur_n in $val_path — leaving GEN unfiltered\n";
         say "GEN: $summary";
      } else {
         my ($clean, $log) = clean_summary(summary => $summary, source => $source, mode => $mode);
         say "GEN: $clean";
         if (@$log) {
            $touched++;
            $total_lits += @$log;
            $by_type{$_->{type}}++ for @$log;
            warn "  ex$cur_n filtered: " . join(', ', map { "$_->{text}\[$_->{type}]" } @$log) . "\n";
         }
      }
   }
   elsif ($line =~ /^REF ?(.*)$/) {
      say "REF: " . ids_to_text($1);
   }
   else {
      say $line;                          # pass through anything else
   }
}

warn "\n---- post-filter summary ($mode) ----\n";
warn "examples:        $examples\n";
warn "examples touched: $touched\n";
warn "literals removed: $total_lits" . (%by_type ? "  (" . join(', ', map { "$_ $by_type{$_}" } sort keys %by_type) . ")" : "") . "\n";
