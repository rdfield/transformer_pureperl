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

my $tok = ML::Tokeniser->new(vocab_size => 8000);
my $emails = read_file("emails.txt");

warn sprintf "read_file: %.2fs\n", time - $t_start;

say "Corpus length: " . length($emails) . " chars";
say "Training tokeniser (7744 merges)...";
$tok->train(text => $emails);
say "Training complete.";
$tok->register_special_tokens({                                                                                                       
      '[PAD]' => 8000,                                                                                                                  
      '[CLS]' => 8001,
      '[UNK]' => 8002,                                                                                                                  
});                                                                                                                                 
                    
$tok->save(file => "client_emails_8000");

