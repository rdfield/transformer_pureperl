#!/usr/bin/perl
# Show THREAD / GENERATED / REFERENCE triples for a random sample, so a human
# can score generations against the SOURCE rather than against the reference.
#
# This matters: the reference summaries are themselves machine-generated, so
# scoring GEN against REF inherits the teacher's errors in both directions —
# a correct generation looks wrong where the teacher was wrong, and a wrong
# one looks right where they agree in error. Only the thread is ground truth.
#
#   perl audit_sample.pl --pairs val.txt --gen gen.txt [--n 20] [--seed 1]
#                        [--vocab model] [--thread-chars 1800]
#
use Modern::Perl;
use Getopt::Long;
use lib '/home/mvine/embedding';
use ML::Tokeniser;
binmode(STDOUT, ':utf8');

my ($pairs, $gen_file) = ('', '');
my $N = 20; my $SEED = 1; my $CHARS = 1800;
my $VOCAB = '/home/mvine/shared/rdf/pretrain_data/cosmo_client_blended2.model';
my $SEP = 16003;
GetOptions('pairs=s' => \$pairs, 'gen=s' => \$gen_file, 'n=i' => \$N,
           'seed=i' => \$SEED, 'vocab=s' => \$VOCAB, 'thread-chars=i' => \$CHARS)
   or die "usage: $0 --pairs val.txt --gen gen.txt [--n 20] [--seed 1]\n";
die "--pairs required\n" unless $pairs && -e $pairs;
die "--gen required\n"   unless $gen_file && -e $gen_file;

my $t = ML::Tokeniser->new(vocab_size => 16000);
$t->load(file => $VOCAB);

# generations, in file order (example i <-> pairs line i)
my (@gen, @ref);
open my $gf, '<', $gen_file or die $!;
while (my $l = <$gf>) {
   push @gen, $1 if $l =~ /^GEN\s*(.*)$/;
   push @ref, $1 if $l =~ /^REF\s*(.*)$/;
}
close $gf;
die "no GEN lines in $gen_file\n" unless @gen;

# only the first scalar(@gen) pairs were generated from
my @lines;
open my $pf, '<', $pairs or die $!;
while (my $l = <$pf>) { last if @lines >= @gen; chomp $l; push @lines, $l; }
close $pf;

srand($SEED);
my @idx = (0 .. $#gen);
for (my $i = $#idx; $i > 0; $i--) { my $j = int rand($i+1); @idx[$i,$j] = @idx[$j,$i]; }
@idx = sort { $a <=> $b } @idx[0 .. ($N > @idx ? $#idx : $N-1)];

my $dec = sub {
   my @i = grep { /^\d+$/ } split /,/, ($_[0] // '');
   return '(empty)' unless @i;
   my $s = $t->decode(ids => \@i);
   $s =~ s/\s+/ /g;
   return $s;
};

say "Scoring $N of ${\ scalar @gen} generations (seed $SEED).";
say "For each: does GENERATED mislead a reader who has seen the THREAD?";
say "Score A=acceptable  B=right gist, wrong detail  C=wrong/garbled\n";

for my $i (@idx) {
   my @ids = grep { /^\d+$/ } split /,/, $lines[$i];
   my ($sep) = grep { $ids[$_] == $SEP } 0 .. $#ids;
   my $thread = defined $sep ? $dec->(join ',', @ids[0 .. $sep-1]) : '(no SEP)';
   $thread = substr($thread, 0, $CHARS) . ' …[truncated]' if length($thread) > $CHARS;

   say '=' x 100;
   printf "EXAMPLE %d   (score: ___ )\n", $i + 1;
   say '=' x 100;
   say "THREAD:\n$thread\n";
   say "GENERATED:\n" . $dec->($gen[$i]) . "\n";
   say "REFERENCE (teacher — also machine-generated, not ground truth):\n" . $dec->($ref[$i]) . "\n";
}
say '=' x 100;
say "Tally: A=__  B=__  C=__   faith+accept = A (+B if a wrong detail is tolerable)";
