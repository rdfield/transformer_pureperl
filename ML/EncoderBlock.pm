package ML::EncoderBlock;

use Modern::Perl;
use ML::ResidualConnection;
use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);

sub type {
   return "ML::EncoderBlock";
}

sub initialise {
   my $self = shift;
   $self->{ln_layer}->initialise();
}

sub optimise {
   my $self = shift;
   $self->{residual_connections}[0]->optimise( @_ );
   $self->{residual_connections}[1]->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   say "ML::EncoderBlock forward called by " . join(", ", caller()) if $self->{debug};
   $self->{input} = dclone($args{batch});

   my $output = [];
   $output = $self->{residual_connections}[0]->forward( batch => $self->{input}, sublayer => $self->{attention_block}, mask => $args{mask} );
   $output = $self->{residual_connections}[1]->forward( batch => $output, sublayer => $self->{feed_forward_block} );

   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{mini_batch_size} ||= $params{mini_batch_size};
 
   my $next = $params{ next };
   
   say "calling residual_connections[1] backward with " . ref($next) if $self->{debug};
   $self->{residual_connections}[1]->backward( next => $next );
   say "calling residual_connections[0] backward with " . ref($self->{residual_connections}[1]) if $self->{debug};
   $self->{residual_connections}[0]->backward( next => $self->{residual_connections}[1] );
   $self->{gradient} = $self->{residual_connections}[0]->gradient();

}

sub gradient {
   my $self = shift;
   return $self->{gradient};
   #return $self->{bn}->gradient(@_);
}

sub outsize {
   my $self = shift;
   return $self->{insize};
}

sub name {
   my $self = shift;
   return $self->{name};
}

sub get_weights {
   my $self = shift;
   return { dropout => $self->{dropout},
            embeddings => $self->{embeddings},
            max_seq_length => $self->{max_seq_length},
            num_heads => $self->{num_heads},
            d_ff => $self->{d_ff},
            attention_block => $self->{attention_block}->get_weights(),
            feed_forward_block => $self->{feed_forward_block}->get_weights()
          };
}

sub new {
   my $class = shift;
   my %args = @_;

   my $embeddings = $args{embeddings};
   die "EncoderBlock invalid embeddings: " . $args{embeddings} unless defined($embeddings) and $embeddings =~ /^\d+$/;
   my $max_seq_length = $args{max_seq_length};
   die "EncoderBlock invalid max_seq_length: " . $args{max_seq_length} unless defined($max_seq_length) and $max_seq_length =~ /^\d+$/;
   my $num_heads = $args{num_heads};
   die "EncoderBlock invalid num_heads: " . $args{num_heads} unless defined($num_heads) and $num_heads =~ /^\d+$/;
   my $d_ff = $args{d_ff};
   die "EncoderBlock invalid d_ff: " . $args{d_ff} unless defined($d_ff) and $d_ff =~ /^\d+$/;
   my $dropout = $args{dropout};
   my $self = {};
   $self->{dropout} = $dropout;
   $self->{embeddings} = $embeddings;
   $self->{max_seq_length} = $max_seq_length;
   $self->{num_heads} = $num_heads;
   $self->{d_ff} = $d_ff;
   $self->{created_by} = "ML::EncoderBlock created by " . join(", ", caller()) . " with max_seq_length = " . $max_seq_length;

   if (defined($args{attention_block})) {
      $self->{attention_block} = ML::MHA->new( %{ $args{ attention_block } } );
   } else {
      $self->{attention_block} = ML::MHA->new( embeddings => $self->{embeddings}, max_seq_length => $self->{max_seq_length}, dropout => $self->{dropout}, num_heads => $self->{num_heads});
   }
   if (defined($args{feed_forward_block})) {
      $self->{feed_forward_block} = ML::PositionwiseFeedForward->new( %{ $args{ feed_forward_block } } );
   } else {
      $self->{feed_forward_block} = ML::PositionwiseFeedForward->new( embeddings => $self->{embeddings}, max_seq_length => $self->{max_seq_length}, creator => $self->{created_by}, dropout => $self->{dropout}, d_ff => $self->{d_ff});
   }

   foreach my $i (0 .. 1) {
      $self->{residual_connections}[$i] = ML::ResidualConnection->new( embeddings => $self->{embeddings}, dropout => $self->{dropout}, max_seq_length => $self->{max_seq_length}, creator => $self->{created_by});
   }
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "encoderblock";
   return bless $self, $class;
}  

1;

__END__

