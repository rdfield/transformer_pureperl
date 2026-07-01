use Modern::Perl;
use lib '.';
use ML::Tokeniser;
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use utf8;

$|++;

my $categories = {
      "OOT" => 6,
      "Onboarding" => 9,
      "Low Margin" => 3,
      "No Action Required" => 2,
      "Order/Delivery" => 7,
      "Billing and Invoice" => 1,
      "Logistic Quote" => 0,
      "Quote Request" => 4,
      "Vendor Quote" => 8,
      "Reports" => 5,
      "Services" => 10
   };

my $filename = "client_emails_8000.model";
open OUTFILE, ">", "tokenised_emails_8000.csv";

my $tok = ML::Tokeniser->new(vocab_size => 8000);
$tok->load(file => "$filename");

my $cls_id = $tok->{special_tokens}{'[CLS]'};

my $csv = Text::CSV->new();

my @lengths;
my %lengths_by_class;
my @buckets;
open my $fhin, "<:encoding(utf8)", "email_categories.csv" or die $!;
my $row_count = 0;
while (my $row = $csv->getline($fhin)) {
   my $body = decode_base64($row->[2]);
   $row_count++;
if ($row_count % 50 == 0) {
   print ".";
}
   utf8::decode($body);
   my @ids = @{$tok->encode(text => $body)};
   my $bucket = int(scalar(@ids)/50);
   if ($bucket > 40) {
      $bucket = 41;
   }
   $buckets[$bucket]++;
   print OUTFILE join(",", $categories->{$row->[1]}, $cls_id, @ids) . "\n";
   push @lengths, scalar(@ids);
   push @{$lengths_by_class{$row->[1]}}, scalar(@ids);
}
say "";
close $fhin;
close OUTFILE;

@lengths = sort { $a <=> $b } @lengths;
my $n = scalar @lengths;
print "Overall: n=$n  median=$lengths[$n/2]  p90=$lengths[int($n*0.9)]  p99=$lengths[int($n*0.99)]  max=$lengths[-1]\n";

foreach my $b(0 .. $#buckets) {
   print "" . ($b * 50) . " - " . (($b + 1) * 50) . ": " . $buckets[$b] . "\n";
}

print "\nPer class:\n";
for my $cls (sort keys %lengths_by_class) {
    my @lens = sort { $a <=> $b } @{$lengths_by_class{$cls}};
    my $n = scalar @lens;
    printf "  %-25s n=%5d  median=%5d  p90=%5d  p99=%5d\n",
        $cls, $n, $lens[$n/2], $lens[int($n*0.9)], $lens[int($n*0.99)];
}

