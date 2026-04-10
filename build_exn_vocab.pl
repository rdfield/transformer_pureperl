#!/usr/bin/perl
use Modern::Perl;
use utf8;
use lib '.';
use ML::Tokeniser;
use File::Temp qw(tempfile);
use File::Slurp;

binmode(STDOUT, ':utf8');

my $tok = ML::Tokeniser->new(vocab_size => 1000);
my $emails = read_file("emails.txt");
say "Corpus length: " . length($emails) . " chars";
say "Training tokeniser (744 merges)...";
$tok->train(text => $emails);
say "Training complete.";
$tok->register_special_tokens({                                                                                                       
      '[PAD]' => 1000,                                                                                                                  
      '[CLS]' => 1001,
      '[UNK]' => 1002,                                                                                                                  
});                                                                                                                                 
                    
$tok->save(file => "exn_emails");

