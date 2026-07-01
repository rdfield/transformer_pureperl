package ML::Tokeniser;
use Modern::Perl;
#use lib '..';
use parent qw(ML::BaseTokeniser);
use Data::Dumper;
use feature 'unicode_strings';
use Time::HiRes qw(time);


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

  my $t_start = time;

   my %params = @_;
   my $text = $params{text};
   die "vocab_size must be >= 256" unless $self->{vocab_size} >= 256;
   my $num_merges = $self->{vocab_size} - 256;

  my $t = time;

   my @chunks = $text =~ /$GPT4_SPLIT_PATTERN/gms;

  warn sprintf "regex split: %.2fs, %d chunks\n", time - $t, scalar(@chunks);

  $t = time;
   my @ids;
   foreach my $ch (@chunks) {
      my @ch_ids = map { utf8::encode($_); unpack('C*', $_)} split//,$ch ;
      push  @ids , \@ch_ids;
   } 
  warn sprintf "byte array : %.2fs\n", time - $t;
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
   $self->{merges} = [];
   my $stats = {};
   my $pair_to_chunks = {};
   $t = time;
=pod
   foreach my $ch (@ids) {
      $stats = $self->get_stats($ch, $stats);
   }
=cut

  for my $ci (0 .. $#ids) {
      my $ch = $ids[$ci];
      for my $i (0 .. $#$ch - 1) {
          my $key = "$ch->[$i],$ch->[$i+1]";
          $stats->{$key}{count}++;
          if ($stats->{$key}{count} == 1) {
             $stats->{$key}{mergeidx} = $self->{mergeidx}++;
          }
          $pair_to_chunks->{$key}{$ci} = 1;
      }
  }

  warn sprintf "stats hash : %.2fs\n", time - $t;
   # module in py resets vocab, but it already set to [0 .. 256]
   foreach my $i (0 .. $num_merges - 1) {
      $t = time;
      if ($i % 50 == 0) {
         print ".";
      }
      my $most_frequent_key;
      foreach my $key (keys %$stats) {
         if (!defined($most_frequent_key)) {
            $most_frequent_key = $key;
         } else {
            if ($stats->{$key}->{count} > $stats->{$most_frequent_key}->{count}
                or ( $stats->{$key}->{count} == $stats->{$most_frequent_key}->{count}
                     and $stats->{$key}->{mergeidx} < $stats->{$most_frequent_key}{mergeidx}) ) {
               $most_frequent_key = $key;
            }
         }
      }
      my $most_frequent_pair = [ split/,/, $most_frequent_key ];
     
      say "pair_to_find = " . join(", ", @$most_frequent_pair) if $self->{verbose};
 warn sprintf "  iter %d: best=%s count=%d  stats{32,256}=%d  stats{32,32}=%d\n",
      $i,
      $most_frequent_key,
      $stats->{$most_frequent_key}->{count},
      ($stats->{"32,256"}{count} // 0),
      ($stats->{"32,32"}{count} // 0);

      my $idx = 256 + $i;
      $t = time;
#      foreach my $ch (@ids) {
      my @affected = keys %{$pair_to_chunks->{$most_frequent_key}};
      if (scalar(@affected) > 0) {
         foreach my $ch (@affected) {
            $ids[$ch] = $self->merge($ids[$ch], $most_frequent_pair, $idx, $stats, $pair_to_chunks, $ch);
         }
      }
  #warn sprintf "merge: %.2fs\n", time - $t;
      #push @{$self->{merges}}, { key => $sorted[0]->{key}, pair => $most_frequent_pair, value => $idx };
      push @{$self->{merges}}, { key => $most_frequent_key, pair => $most_frequent_pair, value => $idx };
      $self->{vocab}[$idx] = $self->{vocab}[$most_frequent_pair->[0]] . $self->{vocab}[$most_frequent_pair->[1]];
      if ($self->{verbose}) {
         say "merge " . ($i+1) . "/$num_merges: (" . join(", ", @$most_frequent_pair) .") -> $idx (" . $self->{vocab}[$idx] . ") had " . $stats->{$most_frequent_key}->{count} . " occurrences";
      }
   }
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
}

sub _build_merges_hash {
   my $self = shift;
   my %h;
   for my $merge (@{$self->{merges}}) {
       $h{ $merge->{key} } = $merge;
   }
   $self->{merges_hash} = \%h;
}

sub _encode {
say "encode called";
   my ($self, $text) = @_;
   my @ids = map { utf8::encode($_); unpack('C*', $_)} split//,$text;
   $self->_build_merges_hash unless exists $self->{merges_hash};

   my $stats = {}; 
   my $pair_to_chunks = {};

my $loop_count = 0;

   while (scalar(@ids) >= 2) {
      my $stats = $self->get_stats(\@ids);
      my $most_frequent_pair;
      my $max_count = -1;

  my $best_mergeidx;

foreach my $pair_key (keys %$stats) {
      if (my $candidate = $self->{merges_hash}{$pair_key}) {
          my $cnt = $stats->{$pair_key}{count};
          my $mi  = $stats->{$pair_key}{mergeidx};

          my $better =
             !defined($most_frequent_pair)        ? 1 :
             $cnt > $max_count                    ? 1 :
             $cnt < $max_count                    ? 0 :
             $mi  < $best_mergeidx                ? 1 :
                                                    0;

         if ($better) {
            $most_frequent_pair = $candidate;
            $max_count = $cnt;
            $best_mergeidx = $mi;
         }
      }
  }

# end of the new minibpe version

      if (!defined($most_frequent_pair)) {
         last;
      }
      my $idx = $most_frequent_pair->{"value"};
      @ids = $self->merge_for_encode(\@ids,  $most_frequent_pair->{pair}  , $idx);#, $stats, $pair_to_chunks, 0);


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
         if (index($text, $tv) >= 0) {
            die "none_raise set, but special token $tv found in text at " . index($text, $tv);
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
