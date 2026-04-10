package ML::BaseTokeniser;
use Modern::Perl;
use Encode qw(encode decode);
use Data::Dumper;
=pod
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
=cut

# import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

sub get_stats {
   my ($self, $ids, $counts) = @_;
=pod
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
=cut
   $counts ||= {};
   if (scalar(%$counts) == 0) {
      $self->{mergeidx} = 0;
   }
   foreach my $i (0 .. $#$ids - 1) {
      my $key = join(",", $ids->[$i] , $ids->[$i + 1] );
      if (defined($counts->{$key})) {
         $counts->{$key}{count}++;
      } else {
         $counts->{$key}{mergeidx} = $self->{mergeidx}++;
         $counts->{$key}{count} = 1;
      }
   }
   return $counts;
}



sub merge {
   my ($self,$ids, $pair, $idx) = @_; 
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

# first two helper functions...
sub replace_control_characters {
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    my ($self, $s) = @_;
    $s =~ s/(\P{XPosixPrint})/sprintf("\\u%04x", ord($1))/eg;
    return $s;
}

sub render_token {
    my ($self, $s) = @_;
    # pretty print a token, escaping control characters
    # s = t.decode('utf-8', errors='replace')
    #s = replace_control_characters(s)
    $s = decode('utf8', $s);
    return $self->replace_control_characters($s);
}

# -----------------------------------------------------------------------------
# the base Tokenizer class

sub new {
   my $proto = shift;
   my $class = ref($proto) || $proto;
   my $self  = {};
   # default: vocab size of 256 (all bytes), no merges, no patterns
   my %params = @_;
   $self->{merges} = [];
   $self->{pattern} = "";
   $self->{special_tokens} = {};
   $self->{mergeidx} = 0;
   bless ($self, $class);
   $self->{vocab} = $self->_build_vocab();
   $self->{vocab_size} = $params{vocab_size};
   return $self;
}

sub train {
   my ($self, $text, $vocab_size, $verbose) = @_;
}

sub token_encode {
   my ($self, $text) = @_;
}

sub token_decode {
   my ($self, $ids) = @_;
}

sub _build_vocab {
   my ($self) = shift;
   my @vocab;
   foreach my $idx (0 .. 255) {
      $vocab[$idx] =  pack("C*",$idx);
   }
   foreach my $m (@{$self->{merges}}) {
      $vocab[$m->{"value"}] = $vocab[$m->{pair}[0]]. $vocab[$m->{pair}[1]];
   }  
=pod
   foreach my $s (keys %{$self->{special_tokens}}) {
      $s->{"key"} =~ /^\((\d+),\s*(\d+)\)$/;
      $vocab[$s->{"value"}] = pack("C*", $1, $2);
   }  
=cut
   %{$self->{inverse_special_tokens}} = reverse %{$self->{special_tokens}};

   return \@vocab;
}

sub save {
   my $self = shift;
   my %params = @_;
   my $file_prefix = $params{file};
=pod
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
=cut
   # write the model: to be used in load() later
   my $model_file_name = $file_prefix . ".model";
   open(F, ">", $model_file_name);
   # write the version, pattern and merges, that's all that's needed
   print F "minbpe v1\n";
   print F $self->{pattern} . "\n";
   # write the special tokens, first the number of them, then each one
   print F scalar(%{$self->{special_tokens}}) . "\n";
   foreach my $st (keys %{$self->{special_tokens}}) {
      print F $st . " " . $self->{special_tokens}{$st} . "\n";
   }
   # the merges dict
   foreach my $m (@{$self->{merges}}) {
      #print F $m->{key} . "\n";
      print F join(" ", @{$m->{pair}}) . "\n";
   }
   # write the vocab: for the human to look at
   my $vocab_file_name = $file_prefix . ".vocab";
   my %inverted_merges = map {$_->{value} => $_->{key}} @{$self->{merges}};
   open(V, ">:encoding(utf-8)", $vocab_file_name);
   foreach my $vocidx (0 .. $#{$self->{vocab}}) {
      # note: many tokens may be partial utf-8 sequences
      # and cannot be decoded into valid strings. Here we're using
      # errors='replace' to replace them with the replacement char �.
      # this also means that we couldn't possibly use .vocab in load()
      # because decoding in this way is a lossy operation!
      my $s = $self->render_token($self->{vocab}[$vocidx]);
      # find the children of this token, if any
      if ($inverted_merges{ $vocidx }) {
         # if this token has children, render it nicely as a merge
         $inverted_merges{ $vocidx } =~ /^(\d+),(\d+)$/;
         my $idx0 = $1;
         my $idx1 = $2;
         my $s0 = $self->render_token($self->{vocab}[$idx0]);
         my $s1 = $self->render_token($self->{vocab}[$idx1]);
         print V "[$s0][$s1] -> [$s] $vocidx\n";
      } else {
         # otherwise this is leaf token, just print it
         # (this should just be the first 256 tokens, the bytes)
         print V "[$s] $vocidx\n";
      }
   }
}

sub load {
   my $self = shift;
   my %params = @_;
   my $model_file = $params{file};

   # """Inverse of save() but only for the model file"""
   return unless $model_file =~ /\.model$/;
   # read the model file
   $self->{merges} = [];
   $self->{special_tokens} = {};
   my $idx = 256;
   open(F, "<:encoding(utf-8)" , $model_file);
   # read the version
   my $version = <F>;
   chomp($version);
   return unless $version eq "minbpe v1";
   # read the pattern
   $self->{pattern} = <F>;
   chomp($self->{pattern});
   # read the special tokens
   my $num_special = <F>;
   chomp($num_special);
say "number of special tokens = " . $num_special;
   $num_special *= 1; 
   foreach my $idx ( 0 .. $num_special -1 ) {
      my $line = <F>;
      chomp($line);
      my ($special, $special_idx) = split/ /, $line;
      $self->{special_tokens}{$special} = $special_idx;
   }
   # read the merges
   while (my $line = <F>) {
      chomp($line);
      my @pair = split / /,$line;
      push @{$self->{merges}}, {pair => \@pair, key => join(",",@pair), value => $idx++ } ;
   }
   $self->{vocab} = $self->_build_vocab();
   close F;
}

sub vocab_size {
   my $self = shift;
   return $self->{vocab_size};
}

1;
