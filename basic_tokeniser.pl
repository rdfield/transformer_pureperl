use Modern::Perl;
use utf8;
use Data::Dumper;
use File::Slurp;

# Step 1: Get the sample text from Nathan Reed's blog post
my $text = q{Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support Unicode" in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception.};

say "Text: $text";
say "Length in characters: " . length($text);

# Step 2: Encode the text to UTF-8 bytes and convert to list of integers
my @tokens =map { utf8::encode($_); unpack('C*', $_)} split//,$text;
say "UTF-8 encoded bytes: ", join(",", @tokens[0..49]);
say "Length in bytes: " . scalar(@tokens);

sub get_stats {
    my @sample_list = @_;
    my %counts = ();
    map { $counts{ "(" . join(", ", $sample_list[$_] , $sample_list[$_ + 1] ) . ")"}++  } 0 .. $#sample_list - 1;
    return \%counts;
}

=pod
# Step 3a: Understand how zip(ids, ids[1:]) works for consecutive pairs
my @sample_list = (1, 2, 3, 4, 5);
my @consecutive_pairs = map { [ $sample_list[$_] , $sample_list[$_ + 1] ] } 0 .. $#sample_list - 1;
say "Sample list: [" . join(",", @sample_list)  . "]";
say "Consecutive pairs:  [" . join(",", map { "(" . join(",", @$_) . ")"  } @consecutive_pairs ) . "]";
=cut

# Step 3: Find the most common consecutive pair using get_stats
my $stats = get_stats(@tokens);
say "Total number of unique pairs: " . scalar(keys %$stats);
my @sorted;
foreach my $key (keys %$stats) {
   push @sorted, { key => $key, count => $stats->{$key} };
}
@sorted = sort { $b->{count} <=> $a->{count} } @sorted;
foreach my $i (0 .. 9) {
   say $sorted[$i]->{key} . ": " . $sorted[$i]->{count} . " times";
}

# Step 4: Get the most frequent pair using max() function
my $most_frequent_pair = $sorted[0]->{key};
say "Most frequent pair: $most_frequent_pair";
say "Occurs " . $sorted[0]->{count} . " times";
# Convert bytes back to characters to see what this pair represents
$most_frequent_pair =~ /^\((\d+),\s*(\d+)\)$/;
my @chars = ( $1 , $2 );
my $char1 = chr($chars[0]);
my $char2 = chr($chars[1]);
say "This represents: '$char1' + '$char2'";

# Step 4a: Verify the most frequent pair by finding its occurrences in the text
my @pair_to_find = @chars; # python is weird and they declare yet another variable for the same thing

# Find all positions where this pair occurs
my @occurrences = ();
foreach my $i (0 .. $#tokens) {
    if ($tokens[$i] == $pair_to_find[0] and $tokens[$i + 1] == $pair_to_find[1]) {
        push @occurrences, $i;
    }
}

say "Found " . scalar(@occurrences) . " occurrences of pair ($pair_to_find[0], $pair_to_find[1])  ('e' + ' ') at positions:";
say "Positions: " . join(",", @occurrences);

# Step 5: Prepare to merge - create new token ID
# Current tokens are 0-255 (256 possible values), so new token will be 256
my $new_token_id = 256;
say "Will replace pair  ($pair_to_find[0], $pair_to_find[1]) with new token ID: $new_token_id";
say "Ready to implement merge function...";

# Step 6: Implement the merge function
sub merge {
   my ($ids, $pair, $idx) = @_;
=pod
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
=cut
   my $newids = [];
   my $i = 0;
   while ($i < scalar(@$ids) ) {
       # if not at the very last position AND the pair matches, replace it
if ($pair->[0] !~ /^\d+$/) {
   die "unexpected pair 0 = " . $pair->[0] ." " . join(",", caller());
}
       if ($ids->[$i] == $pair->[0] and $i < scalar(@$ids) - 1 and $ids->[$i+1] == $pair->[1]) {
           push @$newids, $idx;
           $i += 2;  # skip over the pair
       } else {
           push @$newids, $ids->[$i];
           $i++;
       }
   }
   if (wantarray) {
      return @$newids;
   } else {
      return $newids
   }
}

# Test with simple example
my $test_ids = [5, 6, 6, 7, 9, 1];
my $result = merge($test_ids, [6, 7], 99);
say "Original: " . join(", ", @$test_ids);
say "After merging (6, 7) -> 99: " . join(", ", @$result);


# Step 7: Apply merge to our actual tokens
# Merge the most frequent pair (101, 32) with token ID 256
my $tokens2 = merge(\@tokens, \@pair_to_find, $new_token_id);

say "Original length: " . scalar(@tokens);
say "After merge length: " . scalar(@$tokens2);
say "Reduction: " . ( scalar(@tokens) - scalar(@$tokens2) ) . " tokens";

# Verify the merge worked
say "Occurrences of new token $new_token_id: " . scalar(grep { $_ == $new_token_id } @$tokens2);
my $occurs = 0;
foreach my $i (0 .. $#tokens - 1) {
   $occurs++ if $tokens[$i] == $pair_to_find[0] && $tokens[$i + 1] == $pair_to_find[1];
}
say "Occurrences of old pair in original: $occurs";

# Verify old pair is gone
$occurs = 0;
foreach my $i (0 .. $#$tokens2 - 1) {
   $occurs++ if $tokens2->[$i] == $pair_to_find[0] && $tokens2->[$i + 1] == $pair_to_find[1];
}
say "Occurrences of old pair in new tokens: $occurs";

# Step 8: Iterate the BPE algorithm
# Now we repeat: find most common pair, merge it, repeat...
# Let's do a few more iterations

my $current_tokens = $tokens2;
my $vocab_size = 257;  # Started with 256, now have 257

say "BPE Training Progress:";
say "Step 0: " . scalar(@tokens) . " tokens, vocab size: 256";
say "Step 1: " . scalar(@$current_tokens) . " tokens, vocab size: $vocab_size";

# Do a few more iterations
foreach my $step (2 .. 5) {
   # Find most common pair
   my $stats = get_stats(@$current_tokens);
   if (!scalar(keys %$stats)) {
       last;
   }
   my @sorted;
   foreach my $key (keys %$stats) {
      push @sorted, { key => $key, count => $stats->{$key} };
   }
   @sorted = sort { $b->{count} <=> $a->{count} } @sorted;
   $most_frequent_pair = $sorted[0]->{key};
   $most_frequent_pair =~ /^\((\d+),\s*(\d+)\)$/;
   @pair_to_find = ( $1 , $2 );

    
    # Merge it
    $current_tokens = merge($current_tokens, \@pair_to_find, $vocab_size);
    
    say "Step $step: " . scalar(@$current_tokens) . " tokens, vocab size: " . ($vocab_size + 1);
    say "  Merged pair: $most_frequent_pair -> $vocab_size";
    $vocab_size++;
}

say "Final: " . scalar(@$current_tokens) . " tokens, vocab size: $vocab_size";


$text = read_file('blog_post.txt');

# BPE training
$vocab_size = 276;  # hyperparameter: the desired final vocabulary size
my $num_merges = $vocab_size - 256;
@tokens =map { utf8::decode($_); unpack('C*', $_)} split//,$text;

say "UTF-8 encoded bytes: ", join(", ", @tokens[0..49]);
say "Length in bytes: " . scalar(@tokens);
foreach my $i (0 .. $num_merges - 1) {
   # count up all the pairs
   $stats = get_stats(@tokens);
   # find the pair with the highest count
   my @sorted;
   foreach my $key (keys %$stats) {
      push @sorted, { key => $key, count => $stats->{$key} };
   }
   @sorted = sort { $b->{count} <=> $a->{count} } @sorted;
   $most_frequent_pair = $sorted[0]->{key};
   $most_frequent_pair =~ /^\((\d+),\s*(\d+)\)$/;
   @pair_to_find = ( $1 , $2 );
   # mint a new token: assign it the next available id
   my $idx = 256 + $i;
   # replace all occurrences of pair in tokens with idx
   @tokens = merge(\@tokens, \@pair_to_find, $idx);
    # print progress
   say "merge " . ($i+1) . "/$num_merges: $most_frequent_pair  -> $idx " . $sorted[0]->{count} . " occurrences)";
}

# Track the merges we made
my @merges = (
    {key => "(101, 32)", value => 256},  # 'e' + ' '
    {key => "(100, 32)", value => 257},  # 'd' + ' '
    {key => "(116, 101)", value => 258}, # 't' + 'e'
    {key => "(115, 32)", value => 259},  # 's' + ' '
    {key => "(105, 110)", value =>  260}  # 'i' + 'n'
);
# given ids (list of integers), return Python string
my @vocab;
foreach my $idx (0 .. 255) {
   $vocab[$idx] =  pack("C*",$idx);
}
foreach my $m (@merges) {
    $m->{"key"} =~ /^\((\d+),\s*(\d+)\)$/;
    $vocab[$m->{"value"}] = pack("C*", $1, $2);
}
#say Dumper(\@vocab);
sub decode {
    my $ids = shift;
    # given ids, get tokens
    my $tokens = join("", map { $vocab[$_] } @$ids);
    # convert from bytes to string
    utf8::decode($tokens);
    return $tokens;
}

sub encode {
    my $text = shift;
    # given a string, return list of integers (the tokens)
    my @tokens =  map { utf8::encode($_); unpack('C*', $_)} split//,$text ;
    while (1 == 1) {
       my $stats = get_stats(@tokens);
       if (scalar(@tokens) < 2) {
          last;
       }
       my @sorted;
       foreach my $key (keys %$stats) {
          push @sorted, { key => $key, count => $stats->{$key} };
       }
       @sorted = sort { $b->{count} <=> $a->{count} || $a->{key} cmp $b->{key}} @sorted;
       my $most_frequent_pair;# = $sorted[0]->{key};
       my $max_count = -1;
       foreach my $pair ( @sorted ) {
          if (my ($candidate) = grep { $pair->{key} eq $_->{key} } @merges) {
             if (!defined($most_frequent_pair)) {
                $most_frequent_pair = $candidate;
#say "init most frequest pair " . Dumper($most_frequent_pair);
#say "init max count " . $pair->{count};
                $max_count = $pair->{count};
             } elsif ( $pair->{count} > $max_count) {
                $most_frequent_pair = $candidate;
#say "most frequest pair " . Dumper($most_frequent_pair);
#say "max count " . $pair->{count};
                $max_count = $pair->{count};
             }
          }
       }
       if (!defined($most_frequent_pair)) {
          last;
       }
       my $idx = $most_frequent_pair->{"value"};
   
       $most_frequent_pair->{key} =~ /^\((\d+),\s*(\d+)\)$/;
       my @pair_to_find = ( $1, $2 );
       @tokens = merge(\@tokens,  \@pair_to_find  , $idx)
   }
   if (wantarray) {
      return @tokens;
   } else {
      return \@tokens;
   }
}

say decode([97]);  # Should work fine

say decode([128]);  # This will cause UnicodeDecodeError

say join(",",encode("helloe world!"));

say decode(scalar(encode("helloe world!")));

say join(",",decode(scalar(encode("h"))));

# Test that encode/decode is identity for training text
my $text2 = decode(scalar(encode($text)));
say $text eq $text2;

# Test on new validation text
my $valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard";
say decode(scalar(encode($valtext))) eq $valtext;




