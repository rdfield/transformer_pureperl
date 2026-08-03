use Modern::Perl;

use lib '.';
use ML::Tokeniser;

open my $fhin, "<:encoding(utf8)", "email_summary_ids.txt" or die $!;

my $model = "client_emails_16000.model";

my $tok = ML::Tokeniser->new(vocab_size => 276);
$tok->load(file => "$model");

my $sep_id = $tok->{special_tokens}{"[SEP]"};
die "no [SEP] defined in model" unless $sep_id;

my $processed = 0;
while (my $row = <$fhin>) {
   my @parts = split/,16003,/,$row;
   my $email = $tok->decode(ids => [ split/,/,$parts[0] ]);
say $email;
   my $summary = $tok->decode(ids => [ split/,/,$parts[1] ]);
say $summary;
}




