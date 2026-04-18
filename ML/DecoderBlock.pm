package ML::DecoderBlock;

use Modern::Perl;
use ML::ResidualConnection;
use Data::Dumper;
use Storable qw(dclone);

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);

sub type {
   return "ML::DecoderBlock";
}

sub initialise {
   my $self = shift;
   $self->{ln_layer}->initialise();
}

sub get_grad_tensors {
   my $self = shift;
   return [ map { @{$_->get_grad_tensors()} } @{$self->{residual_connections}} ];
}

sub optimise {
  my $self = shift;
  $self->{residual_connections}[0]->optimise( @_ );
  $self->{residual_connections}[1]->optimise( @_ );
  $self->{residual_connections}[2]->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   say "ML::DecoderBlock forward called by " . join(", ", caller()) if $self->{debug};
   $self->{input} = dclone($args{batch});

   my $output = [];
   $output = $self->{residual_connections}[0]->forward( batch => $self->{input}, sublayer => $self->{self_attention_block}, mask => $args{tgt_mask} );
   $output = $self->{residual_connections}[1]->forward( batch => $output, batch_1 => $args{encoder_output}, batch_2 => $args{encoder_output}, sublayer => $self->{cross_attention_block}, mask => $args{src_mask} );
   $output = $self->{residual_connections}[2]->forward( batch => $output, sublayer => $self->{feed_forward_block});
   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{mini_batch_size} ||= $params{mini_batch_size};
 
   my $next = $params{ next };
   $self->{residual_connections}[2]->backward( next => $next );
   $self->{residual_connections}[1]->backward( next => $self->{residual_connections}[2] );
   $self->{residual_connections}[0]->backward( next => $self->{residual_connections}[1] );
   
   $self->{gradient} = $self->{residual_connections}[0]->gradient();
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub gradient_enc {
   my $self = shift;
   # residual_connections[1] wraps the cross-attention block
   return $self->{residual_connections}[1]->gradient_kv();
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
            tgt_seq_length => $self->{tgt_seq_length},
            num_heads => $self->{num_heads},
            d_ff => $self->{d_ff},
            self_attention_block => $self->{self_attention_block}->get_weights(),
            cross_attention_block => $self->{cross_attention_block}->get_weights(),
            feed_forward_block => $self->{feed_forward_block}->get_weights(),
            residual_connections => [ map { $_->get_weights() } @{$self->{residual_connections}} ],
          };
}

sub new {
   my $class = shift;
   my %args = @_;

   my $embeddings = $args{embeddings};
   die "DecoderBlock invalid embeddings: " . $args{embeddings} unless defined($embeddings) and $embeddings =~ /^\d+$/;
   my $max_seq_length = $args{max_seq_length};
   die "DecoderBlock invalid max_seq_length: " . $args{max_seq_length} unless defined($max_seq_length) and $max_seq_length =~ /^\d+$/;
   my $tgt_seq_length = $args{tgt_seq_length};
   die "DecoderBlock invalid tgt_seq_length: " . $args{tgt_seq_length} unless defined($tgt_seq_length) and $tgt_seq_length =~ /^\d+$/;
   my $num_heads = $args{num_heads};
   die "DecoderBlock invalid num_heads: " . $args{num_heads} unless defined($num_heads) and $num_heads =~ /^\d+$/;
   my $d_ff = $args{d_ff};
   die "DecoderBlock invalid d_ff: " . $args{d_ff} unless defined($d_ff) and $d_ff =~ /^\d+$/;
   my $dropout = $args{dropout};
   die "DecoderBlock invalid dropout: " . $args{dropout} unless defined($dropout) and $dropout =~ /^\d+\.?\d+?$/;
   my $self = {};
   $self->{created_by} = "ML::DecoderBlock created by " . join(", ", caller()) . " with max_seq_length = " . $max_seq_length;
   $self->{dropout} = $dropout;
   $self->{embeddings} = $embeddings;
   $self->{max_seq_length} = $max_seq_length;
   $self->{tgt_seq_length} = $tgt_seq_length;
   $self->{d_ff} = $d_ff;
   $self->{num_heads} = $num_heads;

   if (defined($args{self_attention_block})) {
      $self->{self_attention_block} =  ML::MHA->new( %{ $args{self_attention_block} } );
   } else {
      $self->{self_attention_block} =  ML::MHA->new( embeddings => $self->{embeddings}, 
                                                     max_seq_length => $self->{max_seq_length}, 
                                                     dropout => $self->{dropout}, 
                                                     num_heads => $self->{num_heads} );
   }
   if (defined($args{cross_attention_block})) {
      $self->{cross_attention_block} =  ML::MHA->new( %{ $args{cross_attention_block} } );
   } else {
      $self->{cross_attention_block} =  ML::MHA->new( embeddings => $self->{embeddings}, 
                                                  max_seq_length => $self->{max_seq_length}, 
                                                  dropout => $self->{dropout}, 
                                                  num_heads => $self->{num_heads} );
   }
   if (defined($args{feed_forward_block})) {
      $self->{feed_forward_block} = ML::PositionwiseFeedForward->new( %{ $args{feed_forward_block} } );
   } else {
      $self->{feed_forward_block} = ML::PositionwiseFeedForward->new( embeddings => $self->{embeddings}, 
                                                                   max_seq_length => $self->{max_seq_length}, 
                                                                   creator => $self->{created_by}, 
                                                                   dropout => $self->{dropout}, 
                                                                   d_ff => $self->{d_ff});
   }
   foreach my $i (0 .. 2) {
      $self->{residual_connections}[$i] = ML::ResidualConnection->new( embeddings => $self->{embeddings}, 
                                                                       dropout => $self->{dropout}, 
                                                                       max_seq_length => $self->{max_seq_length}, 
                                                                       creator => $self->{created_by});
   }
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "decoderblock";
   return bless $self, $class;
}  

1;
