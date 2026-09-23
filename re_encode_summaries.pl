use Modern::Perl;
use lib '.';
use ML::Tokeniser;

my $infile = "email_sumararies_regen_v4_shortened.txt";
my $outfile = "email_summaries_8000_v4.txt";


my $tok_16 = ML::Tokeniser->new(vocab_size => 276);
$tok_16->load(file => "client_emails_16000.model");
my $tok_8 = ML::Tokeniser->new(vocab_size => 276);
$tok_8->load(file => "client_emails_8000.model");


open IN, "<", $infile;
open OUT, ">", $outfile;

my $SEP_ID_16 = $tok_16->{special_tokens}{"[SEP]"};
my $SEP_ID_8  = $tok_8->{special_tokens}{"[SEP]"};

my $malformed = 0;

while (my $line = <IN>) {
   chomp $line;
   next unless length $line;
   my @ids = split /,/, $line;
   do { $malformed++; next } if grep { !length } @ids;
   my ($sep_at) = grep { $ids[$_] == $SEP_ID_16 } 0 .. $#ids;
   if (!defined($sep_at) or $sep_at == 0 or $sep_at == $#ids) {
      $malformed++;
      next;
   }
   my @email_ids = @ids[0 .. $sep_at - 1];
   my @summary_ids = @ids[$sep_at + 1 .. $#ids];
#   say "line = $line";
#   say "email_ids";
#   say join(",",@email_ids);
#   say "summary_ids";
#   say join(",",@summary_ids);
   my $in_text = $tok_16->decode(ids => \@email_ids);
   my $in_summary = $tok_16->decode(ids => \@summary_ids);
#   say "in_text = $in_text";
#   say "in_summary = $in_summary";
   my @email_8_ids = @{$tok_8->encode(text => $in_text)};
   my @summary_8_ids = @{$tok_8->encode(text => $in_summary)};
#   say "email_8_ids";
#   say join(",", @email_8_ids);
#   say "summary_8_ids";
#   say join(",", @summary_8_ids);
#   say "sep_8 id = $SEP_ID_8";
   print OUT join(",", @email_8_ids, $SEP_ID_8, @summary_8_ids) . "\n";
}
close IN;
close OUT;

