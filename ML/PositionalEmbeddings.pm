package ML::PositionalEmbeddings;
use lib '/home/mvine/embeddings_mk2';
use Modern::Perl;
use lib '/home/mvine/imagenet';
use ML::Util qw(print_2d_array mult_constant print_1d_array add_2_arrays);
use ML::Embedding;
use Data::Dumper;
use Storable qw(dclone);
use Carp qw(confess);

sub optimise {
   # nothing to do
}

sub get_weights {
   my $self = shift;
   return { embeddings => $self->{embeddings},
            max_seq_length => $self->{max_seq_len},
            dropout => $self->{dropout} };
}

sub new {
   my $class = shift;
   my %args = @_;
   my $self = {};
   $self->{embeddings} = $args{embeddings};
   $self->{max_seq_len} = $args{max_seq_length};
   $self->{dropout} = $args{dropout};
   $self->{pe} = []; # will be seq_len x embeddings
   foreach my $s (0 .. $self->{max_seq_len} - 1) {
      foreach my $d (0 .. $self->{embeddings} - 1) {
         $self->{pe}[$s][$d] = 0;
      }
   }
   $self->{position} = [];
   foreach my $s (0 .. $self->{max_seq_len} - 1) {
       $self->{position}[$s][0] = $s;
   }
   if ($self->{debug}) {
      print_2d_array("position", $self->{position});
   }
   my $div_term =  [];
   my $d = 0;
   while ($d < $self->{embeddings}) {
      push @$div_term,  exp( $d  * ( -log(10000) / $self->{embeddings} ) );
      $d += 2;
   }
   if ($self->{debug}) {
      print_1d_array("div_term", $div_term);
   }
   foreach my $i (0 .. $self->{max_seq_len} - 1) {
      foreach my $j (0 .. $self->{embeddings} - 1) {
         if ($j % 2  == 0) { # even
            my $divisor = 10000**($j / $self->{embeddings});
            $self->{position_embeds}[$i][$j] = sin($i / $divisor);
         } else { # odd
            my $divisor = 10000**(($j - 1) / $self->{embeddings});
            $self->{position_embeds}[$i][$j] = cos($i / $divisor);
         }
      }
   }
   if ($self->{debug}) {
      print_2d_array("position embeds", $self->{position_embeds});
   }
   $self->{position_embeds} = [ $self->{position_embeds} ]; # so that the dimensions match later calculations

   return bless $self, $class;
}

sub backward {
   my $self = shift;
   my %args = @_;
   $self->{gradient} = $args{next}->{gradient};
}

sub gradient {
   my $self = shift;
   my %args = @_;
   if ($args{gradient}) { # testing, will be $args{next}->{gradient};
      $self->{gradient} = dclone($args{gradient});
   }
   return $self->{gradient};
}

sub forward {
   my $self = shift;
   my %args = @_;
   my $input = dclone($args{batch});
   my $batch_size = scalar(@$input);
   my $seq_len = scalar(@{$input->[0]});
   my $batch_embeddings = scalar(@{$input->[0][0]});
   my $output = [];
   foreach my $b (0 .. $batch_size - 1) {
      $output->[$b] = add_2_arrays($input->[$b], $self->{position_embeds}[0]);
      foreach my $i (0 .. $seq_len - 1) {
         foreach my $j (0 .. $self->{embeddings} - 1) {
            $output->[$b][$i][$j] = 0 if rand() < $self->{dropout};
         }
      }
   }

   return $output;
}

1;
