package ML::Tokeniser;
use Modern::Perl;
#use lib '..';
use parent qw(ML::BaseTokeniser);
use Data::Dumper;
use feature 'unicode_strings';

#our $GPT4_SPLIT_PATTERN = qr/(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/u;
our $GPT4_SPLIT_PATTERN =  qr/'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/;

sub new {
   my $class = shift;
   my $self = $class->SUPER::new(@_);
   $self->{special_tokens} = {};
   $self->{inverse_special_tokens} = {};
die unless $self->{vocab_size} =~ /^\d+$/;
   return $self;
}

sub merges {
   my $self = shift;
   return $self->{merges};
}

sub train {
   my $self = shift;
   my %params = @_;
   my $text = $params{text};
   die "vocab_size must be >= 256" unless $self->{vocab_size} >= 256;
   my $num_merges = $self->{vocab_size} - 256;
   my @chunks = $text =~ /$GPT4_SPLIT_PATTERN/gms;
   my @ids;
   foreach my $ch (@chunks) {
      my @ch_ids = map { utf8::encode($_); unpack('C*', $_)} split//,$ch ;
      push  @ids , \@ch_ids;
   } 
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
   $self->{merges} = [];
   # module in py resets vocab, but it already set to [0 .. 256]
   foreach my $i (0 .. $num_merges - 1) {
      my $stats = {};
      foreach my $ch (@ids) {
         $stats = $self->get_stats($ch, $stats);
      }
      my @sorted;
      foreach my $key (keys %$stats) {
         push @sorted, { key => $key, pairs => [ split/,/,$key], count => $stats->{$key}{count}, mergeidx => $stats->{$key}{mergeidx} };
      }
      #@sorted = sort { $b->{count} <=> $a->{count} || $a->{pairs}[0] <=> $b->{pairs}[0] || $a->{pairs}[1] <=> $b->{pairs}[1]} @sorted;
      @sorted = sort { $b->{count} <=> $a->{count} || $a->{mergeidx} <=> $b->{mergeidx}} @sorted;
      my $most_frequent_pair = $sorted[0]->{pairs};
      say "pair_to_find = " . join(", ", @$most_frequent_pair) if $self->{verbose};
      my $idx = 256 + $i;
      foreach my $ch (@ids) {
         $ch = $self->merge($ch, $most_frequent_pair, $idx);
      }
      push @{$self->{merges}}, { key => $sorted[0]->{key}, pair => $most_frequent_pair, value => $idx };
      $self->{vocab}[$idx] = $self->{vocab}[$most_frequent_pair->[0]] . $self->{vocab}[$most_frequent_pair->[1]];
      if ($self->{verbose}) {
         say "merge " . ($i+1) . "/$num_merges: (" . join(", ", @$most_frequent_pair) .") -> $idx (" . $self->{vocab}[$idx] . ") had " . $sorted[0]->{count} . " occurrences";
      }
   }
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
}


sub _encode {
   my ($self, $text) = @_;
#   my @ids = map { utf8::decode($_); unpack('C*', $_)} split//,$text;
   my @ids = map { utf8::encode($_); unpack('C*', $_)} split//,$text;
   while (scalar(@ids) >= 2) {
      my $stats = $self->get_stats(\@ids);
      my @sorted;
      foreach my $key (keys %$stats) {
         push @sorted, { key => $key, pairs => [ split/,/,$key], count => $stats->{$key}{count} };
      }
# this sort is for the minimum
      @sorted = sort { $b->{count} <=> $a->{count} || $b->{pairs}[0] <=> $a->{pairs}[0] || $b->{pairs}[1] <=> $a->{pairs}[1]} @sorted;
      my $most_frequent_pair;
      my $max_count = -1;
      foreach my $pair ( @sorted ) {
         if (my ($candidate) = grep { $pair->{key} eq $_->{key} } @{$self->{merges}}) {
            if (!defined($most_frequent_pair)) {
               $most_frequent_pair = $candidate;
               $max_count = $pair->{count};
            } elsif ( $pair->{count} > $max_count) {
               $most_frequent_pair = $candidate;
               $max_count = $pair->{count};
            }
         }
      }
      if (!defined($most_frequent_pair)) {
         last;
      }
      my $idx = $most_frequent_pair->{"value"};
    
      @ids = $self->merge(\@ids,  $most_frequent_pair->{pair}  , $idx)

   }
   return @ids;
}

sub encode {
   my ($self) = shift;
   my %params = @_;
   my ($text) = $params{text} or die;
   my %options = @_;
   my $special = {};
   $options{allowed_special} ||= "none_raise";
   my $allowed_special = $options{allowed_special};
   if ($allowed_special eq "all") {
      $special = $self->{special_tokens};
   } elsif ( $allowed_special eq "none" ) {
   } elsif ( $allowed_special eq "none_raise" ) {
      foreach my $tv (keys %{$self->{special_tokens}}) {
         if (index($text, $tv)) {
            die "none_raise set, but special token $tv found in text";
         }
      }
   } elsif ( ref($allowed_special) eq "ARRAY" ) {
      foreach my $st (keys %{$self->{special_tokens}}) {
         if ( grep { $st eq $_ } @$allowed_special ) {
            $special->{ $st } = $self->{special_tokens}{$st};
         }
      }
   } else {
      die "unknown value for 'allowed_special'";
   }
   if (!scalar(%$special)) {
      my @result = $self->_encode($text);
      return wantarray ? @result : \@result;
   }
   my $split_text = "(" . join('|', map { quotemeta($_) } keys %$special ) . ')';
   my $split_re = qr{$split_text};
   my @chunks = split $split_re, $text;
   my @ids = ();
   foreach my $ch (@chunks) {
      if (grep {$ch eq $_} keys %$special) {
          push @ids, $special->{$ch};
      } else {
          push @ids, $self->_encode($ch);
      }
   }
   if (wantarray) {
      return @ids;
   } else {
      return \@ids;
   }
}

sub decode {
   my ($self) = shift;
   my %params = @_;
   my @ids = @{$params{ids}};
   my $tokens = "";
   foreach my $id (@ids) {
      if (grep{ $id eq $_ } keys %{$self->{inverse_special_tokens}}) {
         $tokens .= $self->{inverse_special_tokens}{$id};
      } else {
         $tokens .= $self->{vocab}[$id];
      }
   }
   utf8::decode($tokens);
   return $tokens;
}

sub register_special_tokens {
   my ($self, $special_tokens) = @_;
   # special_tokens is a dictionary of str -> int
   # example: {"<|endoftext|>": 100257}
   $self->{special_tokens} = $special_tokens;
   %{$self->{inverse_special_tokens}} = reverse %{$self->{special_tokens}};
}

1;
