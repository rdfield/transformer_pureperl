#!/usr/bin/perl
use Modern::Perl;
use utf8;
use lib '.';
use ML::Tokeniser;
use File::Temp qw(tempfile);
use File::Slurp;
use Time::HiRes qw(time);

binmode(STDOUT, ':utf8');
$|++;

my $t_start = time;

my ($vocab_size, $infile, $outfile) = @ARGV;
die "usage perl build_client_vocab.pl <vocab size> <train data file name> <output file name>" unless (defined($vocab_size) and defined($infile) and -e $infile and defined($outfile));

my $tok = ML::Tokeniser->new(vocab_size => $vocab_size);
my $emails = read_file($infile, binmode => ":utf8");   # train() takes decoded text

$tok->train(text => $emails);
$tok->register_special_tokens({
      '[PAD]' => $vocab_size,
      '[CLS]' => $vocab_size + 1,
      '[UNK]' => $vocab_size + 2,
      '[SEP]' => $vocab_size + 3,   # SLM: email/summary separator
      '[EOS]' => $vocab_size + 4,   # SLM: end of generated summary
});                                                                                                                                 
                    
$tok->save(file => $outfile);

