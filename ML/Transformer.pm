use Modern::Perl;
package ML::Transformer;
use ML::LossGradient;
use ML::Util qw(global_clip_grad_norm);
use Storable qw(dclone);
use Data::Dumper;
use JSON;
use File::Slurp;

sub new {
   my $class = shift;
   my %self = @_;
   # expecting encoder, decoder, sec_embed, tgt_embed, src_pos, tgt_pos, projection_layer
   return bless \%self, $class;
}

sub encode {
   my $self = shift;
   my %args = @_;
   # expecting src, src_mask
   my $src = $self->{src_embed}->forward( batch => $args{src} );
   $src = $self->{src_pos}->forward( batch => $src );
   return $self->{encoder}->forward( batch => $src, mask => $args{src_mask} );
}

sub decode {
   my $self = shift;
   my %args = @_;
   # expecting encoder_output, src_mask, tgt, tgt_mask
   my $tgt = $self->{tgt_embed}->forward( batch => $args{tgt} );
   $tgt = $self->{tgt_pos}->forward( batch => $tgt );
   return $self->{decoder}->forward( batch => $tgt, encoder_output => $args{encoder_output}, src_mask => $args{src_mask}, tgt_mask => $args{tgt_mask});
}

sub project {
   my $self = shift;
   my %args = @_;
   #my $batch_size = scalar(@{$args{batch}});
   #my $output = [];
   #foreach my $b (0 .. $batch_size - 1) {
   #   $output->[$b] = $self->{projection_layer}->forward( batch => $args{batch}->[$b] );
   #}
   my $output = $self->{projection_layer}->forward( batch => $args{batch} );
   return $output;
}

sub update {
   my $self = shift;
   my %args = @_;
   my $next = ML::LossGradient->new( gradient => $args{gradient} ); 
   if ($args{projection}) {
      $self->{projection_layer}->backward( next => $next );
      $next = $self->{projection_layer};
   }
   if ($args{decode}) {
      $self->{decoder}->backward( next => $next );
      $self->{tgt_pos}->backward( next => $self->{decoder} );
      $self->{tgt_embed}->backward( next => $self->{tgt_pos} );
      $next = $self->{tgt_embed};
   }
   if ($args{encode}) {
      my $enc_grad   = $self->{decoder}->gradient_enc();
      my $enc_source = defined($enc_grad)
          ? ML::LossGradient->new(gradient => $enc_grad)
          : $next;   # fallback when decode block was not run
      $self->{encoder}->backward( next => $enc_source );
      $self->{src_pos}->backward( next => $self->{encoder} );
      $self->{src_embed}->backward( next => $self->{src_pos} );
      $next = $self->{src_embed};
   }
   if (defined $args{max_norm}) {
      my $tensors = $self->collect_grad_tensors( %args );
      $self->{last_grad_norm} = global_clip_grad_norm($tensors, $args{max_norm});
   }
   if ($args{projection}) {
      $self->{projection_layer}->optimise( learning_rate => $args{learning_rate} );
   }
   if ($args{decode}) {
      $self->{decoder}->optimise( learning_rate => $args{learning_rate} );
      $self->{tgt_pos}->optimise( learning_rate => $args{learning_rate} );
      $self->{tgt_embed}->optimise( learning_rate => $args{learning_rate} );
   }
   if ($args{encode}) {
      $self->{encoder}->optimise( learning_rate => $args{learning_rate} );
      $self->{src_pos}->optimise( learning_rate => $args{learning_rate} );
      $self->{src_embed}->optimise( learning_rate => $args{learning_rate} );
   }
   
}
      
sub collect_grad_tensors {
   my $self = shift;
   my %args = @_;
   my @t;
   push @t, @{$self->{projection_layer}->get_grad_tensors()} if $args{projection};
   push @t, @{$self->{decoder}->get_grad_tensors()},
            @{$self->{tgt_embed}->get_grad_tensors()}         if $args{decode};
   push @t, @{$self->{encoder}->get_grad_tensors()},
            @{$self->{src_embed}->get_grad_tensors()}         if $args{encode};
   return \@t;
}

sub save_model {
   my $self = shift;
   my %args = @_;
   die "no filename parameter" unless defined($args{filename});
   open(my $FH, ">:encoding(utf-8)", $args{filename}) or die $!;
   my $trained_model = { src_embed => $self->{src_embed}->get_weights() } ;
   $trained_model->{tgt_embed} = $self->{tgt_embed}->get_weights();
   $trained_model->{src_pos} = $self->{src_pos}->get_weights();
   $trained_model->{tgt_pos} = $self->{tgt_pos}->get_weights();
   $trained_model->{encoder} = $self->{encoder}->get_weights();
   $trained_model->{decoder} = $self->{decoder}->get_weights();
   $trained_model->{projection_layer} = $self->{projection_layer}->get_weights();
   print $FH to_json($trained_model);
   close $FH
}

sub load_model {
   my $class = shift;
   my %args = @_;
   die "no filename parameter" unless defined($args{filename});
   my $json_data = read_file( $args{filename} ) or die $!;
   my $trained_params = from_json( $json_data );
   my $trained_model = {};
   $trained_model->{src_embed} = ML::InputEmbeddings->new( %{$trained_params->{src_embed}} );
   $trained_model->{tgt_embed} = ML::InputEmbeddings->new( %{$trained_params->{tgt_embed}} );
   $trained_model->{src_pos} = ML::PositionalEmbeddings->new( %{$trained_params->{src_pos}} );
   $trained_model->{tgt_pos} = ML::PositionalEmbeddings->new( %{$trained_params->{tgt_pos}} );
   $trained_model->{encoder} = ML::Encoder->new( %{$trained_params->{encoder}} );
   $trained_model->{decoder} = ML::Decoder->new( %{$trained_params->{decoder}} );
   $trained_model->{projection_layer} = ML::Linear->new( insize => $trained_params->{projection_layer}{insize}, outsize => $trained_params->{projection_layer}{outsize} );
   $trained_model->{projection_layer}->set_weights_and_biases( weights => $trained_params->{projection_layer}{weights}, biases => $trained_params->{projection_layer}{biases} );
   return bless $trained_model, $class;
}

1;

   
