ст
Ѓэ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ЇГ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Ж
SGD/m/output-layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/m/output-layer/bias

+SGD/m/output-layer/bias/Read/ReadVariableOpReadVariableOpSGD/m/output-layer/bias*
_output_shapes
:*
dtype0
О
SGD/m/output-layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
**
shared_nameSGD/m/output-layer/kernel
З
-SGD/m/output-layer/kernel/Read/ReadVariableOpReadVariableOpSGD/m/output-layer/kernel*
_output_shapes

:
*
dtype0
Д
SGD/m/input-layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameSGD/m/input-layer/bias
}
*SGD/m/input-layer/bias/Read/ReadVariableOpReadVariableOpSGD/m/input-layer/bias*
_output_shapes
:
*
dtype0
М
SGD/m/input-layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*)
shared_nameSGD/m/input-layer/kernel
Е
,SGD/m/input-layer/kernel/Read/ReadVariableOpReadVariableOpSGD/m/input-layer/kernel*
_output_shapes

:
*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
output-layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput-layer/bias
s
%output-layer/bias/Read/ReadVariableOpReadVariableOpoutput-layer/bias*
_output_shapes
:*
dtype0
В
output-layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameoutput-layer/kernel
{
'output-layer/kernel/Read/ReadVariableOpReadVariableOpoutput-layer/kernel*
_output_shapes

:
*
dtype0
x
input-layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameinput-layer/bias
q
$input-layer/bias/Read/ReadVariableOpReadVariableOpinput-layer/bias*
_output_shapes
:
*
dtype0
А
input-layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*#
shared_nameinput-layer/kernel
y
&input-layer/kernel/Read/ReadVariableOpReadVariableOpinput-layer/kernel*
_output_shapes

:
*
dtype0
Д
!serving_default_input-layer_inputPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
С
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input-layer_inputinput-layer/kernelinput-layer/biasoutput-layer/kerneloutput-layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_10721

NoOpNoOp
л
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶
valueЬBЩ BТ
Ъ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
∞
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
!trace_0
"trace_1
#trace_2
$trace_3* 
6
%trace_0
&trace_1
'trace_2
(trace_3* 
* 
o
)
_variables
*_iterations
+_learning_rate
,_index_dict
-	momentums
._update_step_xla*

/serving_default* 

0
1*

0
1*
* 
У
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

5trace_0* 

6trace_0* 
b\
VARIABLE_VALUEinput-layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEinput-layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

<trace_0* 

=trace_0* 
c]
VARIABLE_VALUEoutput-layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput-layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

>0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
'
*0
@1
A2
B3
C4*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
@0
A1
B2
C3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
D	variables
E	keras_api
	Ftotal
	Gcount*
H
H	variables
I	keras_api
	Jtotal
	Kcount
L
_fn_kwargs*
c]
VARIABLE_VALUESGD/m/input-layer/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUESGD/m/input-layer/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUESGD/m/output-layer/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/output-layer/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

D	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

J0
K1*

H	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input-layer/kernel/Read/ReadVariableOp$input-layer/bias/Read/ReadVariableOp'output-layer/kernel/Read/ReadVariableOp%output-layer/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp,SGD/m/input-layer/kernel/Read/ReadVariableOp*SGD/m/input-layer/bias/Read/ReadVariableOp-SGD/m/output-layer/kernel/Read/ReadVariableOp+SGD/m/output-layer/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_10882
С
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput-layer/kernelinput-layer/biasoutput-layer/kerneloutput-layer/bias	iterationlearning_rateSGD/m/input-layer/kernelSGD/m/input-layer/biasSGD/m/output-layer/kernelSGD/m/output-layer/biastotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_10934ћЉ
®
’
2__inference_1-input-10-opt-mae_layer_call_fn_10734

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10592o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы&
Л
__inference__traced_save_10882
file_prefix1
-savev2_input_layer_kernel_read_readvariableop/
+savev2_input_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop7
3savev2_sgd_m_input_layer_kernel_read_readvariableop5
1savev2_sgd_m_input_layer_bias_read_readvariableop8
4savev2_sgd_m_output_layer_kernel_read_readvariableop6
2savev2_sgd_m_output_layer_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: с
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ
valueРBНB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЛ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ќ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop3savev2_sgd_m_input_layer_kernel_read_readvariableop1savev2_sgd_m_input_layer_bias_read_readvariableop4savev2_sgd_m_output_layer_kernel_read_readvariableop2savev2_sgd_m_output_layer_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*c
_input_shapesR
P: :
:
:
:: : :
:
:
:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$	 

_output_shapes

:
: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∆
Ш
+__inference_input-layer_layer_call_fn_10788

inputs
unknown:

	unknown_0:

identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_input-layer_layer_call_and_return_conditional_losses_10569o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…
а
2__inference_1-input-10-opt-mae_layer_call_fn_10603
input_layer_input
unknown:

	unknown_0:

	unknown_1:

	unknown_2:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10592o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
…	
ч
F__inference_input-layer_layer_call_and_return_conditional_losses_10569

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
д
«
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10592

inputs#
input_layer_10570:

input_layer_10572:
$
output_layer_10586:
 
output_layer_10588:
identityИҐ#input-layer/StatefulPartitionedCallҐ$output-layer/StatefulPartitionedCallщ
#input-layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_10570input_layer_10572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_input-layer_layer_call_and_return_conditional_losses_10569£
$output-layer/StatefulPartitionedCallStatefulPartitionedCall,input-layer/StatefulPartitionedCall:output:0output_layer_10586output_layer_10588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output-layer_layer_call_and_return_conditional_losses_10585|
IdentityIdentity-output-layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
NoOpNoOp$^input-layer/StatefulPartitionedCall%^output-layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2J
#input-layer/StatefulPartitionedCall#input-layer/StatefulPartitionedCall2L
$output-layer/StatefulPartitionedCall$output-layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Н
—
#__inference_signature_wrapper_10721
input_layer_input
unknown:

	unknown_0:

	unknown_1:

	unknown_2:
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_10552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
 	
ш
G__inference_output-layer_layer_call_and_return_conditional_losses_10817

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
д
«
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10652

inputs#
input_layer_10641:

input_layer_10643:
$
output_layer_10646:
 
output_layer_10648:
identityИҐ#input-layer/StatefulPartitionedCallҐ$output-layer/StatefulPartitionedCallщ
#input-layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_10641input_layer_10643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_input-layer_layer_call_and_return_conditional_losses_10569£
$output-layer/StatefulPartitionedCallStatefulPartitionedCall,input-layer/StatefulPartitionedCall:output:0output_layer_10646output_layer_10648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output-layer_layer_call_and_return_conditional_losses_10585|
IdentityIdentity-output-layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
NoOpNoOp$^input-layer/StatefulPartitionedCall%^output-layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2J
#input-layer/StatefulPartitionedCall#input-layer/StatefulPartitionedCall2L
$output-layer/StatefulPartitionedCall$output-layer/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…
а
2__inference_1-input-10-opt-mae_layer_call_fn_10676
input_layer_input
unknown:

	unknown_0:

	unknown_1:

	unknown_2:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
Е
“
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10690
input_layer_input#
input_layer_10679:

input_layer_10681:
$
output_layer_10684:
 
output_layer_10686:
identityИҐ#input-layer/StatefulPartitionedCallҐ$output-layer/StatefulPartitionedCallД
#input-layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_10679input_layer_10681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_input-layer_layer_call_and_return_conditional_losses_10569£
$output-layer/StatefulPartitionedCallStatefulPartitionedCall,input-layer/StatefulPartitionedCall:output:0output_layer_10684output_layer_10686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output-layer_layer_call_and_return_conditional_losses_10585|
IdentityIdentity-output-layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
NoOpNoOp$^input-layer/StatefulPartitionedCall%^output-layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2J
#input-layer/StatefulPartitionedCall#input-layer/StatefulPartitionedCall2L
$output-layer/StatefulPartitionedCall$output-layer/StatefulPartitionedCall:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
Е
“
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10704
input_layer_input#
input_layer_10693:

input_layer_10695:
$
output_layer_10698:
 
output_layer_10700:
identityИҐ#input-layer/StatefulPartitionedCallҐ$output-layer/StatefulPartitionedCallД
#input-layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_10693input_layer_10695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_input-layer_layer_call_and_return_conditional_losses_10569£
$output-layer/StatefulPartitionedCallStatefulPartitionedCall,input-layer/StatefulPartitionedCall:output:0output_layer_10698output_layer_10700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output-layer_layer_call_and_return_conditional_losses_10585|
IdentityIdentity-output-layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€У
NoOpNoOp$^input-layer/StatefulPartitionedCall%^output-layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2J
#input-layer/StatefulPartitionedCall#input-layer/StatefulPartitionedCall2L
$output-layer/StatefulPartitionedCall$output-layer/StatefulPartitionedCall:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
 	
ш
G__inference_output-layer_layer_call_and_return_conditional_losses_10585

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs
Ќ
ф
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10763

inputs<
*input_layer_matmul_readvariableop_resource:
9
+input_layer_biasadd_readvariableop_resource:
=
+output_layer_matmul_readvariableop_resource:
:
,output_layer_biasadd_readvariableop_resource:
identityИҐ"input-layer/BiasAdd/ReadVariableOpҐ!input-layer/MatMul/ReadVariableOpҐ#output-layer/BiasAdd/ReadVariableOpҐ"output-layer/MatMul/ReadVariableOpМ
!input-layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Б
input-layer/MatMulMatMulinputs)input-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
К
"input-layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ъ
input-layer/BiasAddBiasAddinput-layer/MatMul:product:0*input-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
О
"output-layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Щ
output-layer/MatMulMatMulinput-layer/BiasAdd:output:0*output-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€М
#output-layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
output-layer/BiasAddBiasAddoutput-layer/MatMul:product:0+output-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentityoutput-layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Џ
NoOpNoOp#^input-layer/BiasAdd/ReadVariableOp"^input-layer/MatMul/ReadVariableOp$^output-layer/BiasAdd/ReadVariableOp#^output-layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"input-layer/BiasAdd/ReadVariableOp"input-layer/BiasAdd/ReadVariableOp2F
!input-layer/MatMul/ReadVariableOp!input-layer/MatMul/ReadVariableOp2J
#output-layer/BiasAdd/ReadVariableOp#output-layer/BiasAdd/ReadVariableOp2H
"output-layer/MatMul/ReadVariableOp"output-layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
©
в
 __inference__wrapped_model_10552
input_layer_inputM
;input_10_opt_mae_input_layer_matmul_readvariableop_resource:
J
<input_10_opt_mae_input_layer_biasadd_readvariableop_resource:
N
<input_10_opt_mae_output_layer_matmul_readvariableop_resource:
K
=input_10_opt_mae_output_layer_biasadd_readvariableop_resource:
identityИҐ51-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOpҐ41-input-10-opt-mae/input-layer/MatMul/ReadVariableOpҐ61-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOpҐ51-input-10-opt-mae/output-layer/MatMul/ReadVariableOp∞
41-input-10-opt-mae/input-layer/MatMul/ReadVariableOpReadVariableOp;input_10_opt_mae_input_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0≤
%1-input-10-opt-mae/input-layer/MatMulMatMulinput_layer_input<1-input-10-opt-mae/input-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
Ѓ
51-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOpReadVariableOp<input_10_opt_mae_input_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0”
&1-input-10-opt-mae/input-layer/BiasAddBiasAdd/1-input-10-opt-mae/input-layer/MatMul:product:0=1-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
≤
51-input-10-opt-mae/output-layer/MatMul/ReadVariableOpReadVariableOp<input_10_opt_mae_output_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0“
&1-input-10-opt-mae/output-layer/MatMulMatMul/1-input-10-opt-mae/input-layer/BiasAdd:output:0=1-input-10-opt-mae/output-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€∞
61-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOpReadVariableOp=input_10_opt_mae_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0÷
'1-input-10-opt-mae/output-layer/BiasAddBiasAdd01-input-10-opt-mae/output-layer/MatMul:product:0>1-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
IdentityIdentity01-input-10-opt-mae/output-layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€¶
NoOpNoOp6^1-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOp5^1-input-10-opt-mae/input-layer/MatMul/ReadVariableOp7^1-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOp6^1-input-10-opt-mae/output-layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2n
51-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOp51-input-10-opt-mae/input-layer/BiasAdd/ReadVariableOp2l
41-input-10-opt-mae/input-layer/MatMul/ReadVariableOp41-input-10-opt-mae/input-layer/MatMul/ReadVariableOp2p
61-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOp61-input-10-opt-mae/output-layer/BiasAdd/ReadVariableOp2n
51-input-10-opt-mae/output-layer/MatMul/ReadVariableOp51-input-10-opt-mae/output-layer/MatMul/ReadVariableOp:Z V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinput-layer_input
љ>
¶
!__inference__traced_restore_10934
file_prefix5
#assignvariableop_input_layer_kernel:
1
#assignvariableop_1_input_layer_bias:
8
&assignvariableop_2_output_layer_kernel:
2
$assignvariableop_3_output_layer_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: =
+assignvariableop_6_sgd_m_input_layer_kernel:
7
)assignvariableop_7_sgd_m_input_layer_bias:
>
,assignvariableop_8_sgd_m_output_layer_kernel:
8
*assignvariableop_9_sgd_m_output_layer_bias:%
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: 
identity_15ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ъ
valueРBНB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHО
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B й
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOpAssignVariableOp#assignvariableop_input_layer_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:≥
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_6AssignVariableOp+assignvariableop_6_sgd_m_input_layer_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_7AssignVariableOp)assignvariableop_7_sgd_m_input_layer_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_8AssignVariableOp,assignvariableop_8_sgd_m_output_layer_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_9AssignVariableOp*assignvariableop_9_sgd_m_output_layer_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ќ
ф
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10779

inputs<
*input_layer_matmul_readvariableop_resource:
9
+input_layer_biasadd_readvariableop_resource:
=
+output_layer_matmul_readvariableop_resource:
:
,output_layer_biasadd_readvariableop_resource:
identityИҐ"input-layer/BiasAdd/ReadVariableOpҐ!input-layer/MatMul/ReadVariableOpҐ#output-layer/BiasAdd/ReadVariableOpҐ"output-layer/MatMul/ReadVariableOpМ
!input-layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Б
input-layer/MatMulMatMulinputs)input-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
К
"input-layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ъ
input-layer/BiasAddBiasAddinput-layer/MatMul:product:0*input-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
О
"output-layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Щ
output-layer/MatMulMatMulinput-layer/BiasAdd:output:0*output-layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€М
#output-layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
output-layer/BiasAddBiasAddoutput-layer/MatMul:product:0+output-layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentityoutput-layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Џ
NoOpNoOp#^input-layer/BiasAdd/ReadVariableOp"^input-layer/MatMul/ReadVariableOp$^output-layer/BiasAdd/ReadVariableOp#^output-layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2H
"input-layer/BiasAdd/ReadVariableOp"input-layer/BiasAdd/ReadVariableOp2F
!input-layer/MatMul/ReadVariableOp!input-layer/MatMul/ReadVariableOp2J
#output-layer/BiasAdd/ReadVariableOp#output-layer/BiasAdd/ReadVariableOp2H
"output-layer/MatMul/ReadVariableOp"output-layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…	
ч
F__inference_input-layer_layer_call_and_return_conditional_losses_10798

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®
’
2__inference_1-input-10-opt-mae_layer_call_fn_10747

inputs
unknown:

	unknown_0:

	unknown_1:

	unknown_2:
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *V
fQRO
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10652o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
Щ
,__inference_output-layer_layer_call_fn_10807

inputs
unknown:

	unknown_0:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output-layer_layer_call_and_return_conditional_losses_10585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€

 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*√
serving_defaultѓ
O
input-layer_input:
#serving_default_input-layer_input:0€€€€€€€€€@
output-layer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:К_
і
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
!trace_0
"trace_1
#trace_2
$trace_32Т
2__inference_1-input-10-opt-mae_layer_call_fn_10603
2__inference_1-input-10-opt-mae_layer_call_fn_10734
2__inference_1-input-10-opt-mae_layer_call_fn_10747
2__inference_1-input-10-opt-mae_layer_call_fn_10676њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z!trace_0z"trace_1z#trace_2z$trace_3
й
%trace_0
&trace_1
'trace_2
(trace_32ю
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10763
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10779
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10690
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10704њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z%trace_0z&trace_1z'trace_2z(trace_3
’B“
 __inference__wrapped_model_10552input-layer_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
К
)
_variables
*_iterations
+_learning_rate
,_index_dict
-	momentums
._update_step_xla"
experimentalOptimizer
,
/serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
п
5trace_02“
+__inference_input-layer_layer_call_fn_10788Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z5trace_0
К
6trace_02н
F__inference_input-layer_layer_call_and_return_conditional_losses_10798Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z6trace_0
$:"
2input-layer/kernel
:
2input-layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
р
<trace_02”
,__inference_output-layer_layer_call_fn_10807Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z<trace_0
Л
=trace_02о
G__inference_output-layer_layer_call_and_return_conditional_losses_10817Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z=trace_0
%:#
2output-layer/kernel
:2output-layer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ОBЛ
2__inference_1-input-10-opt-mae_layer_call_fn_10603input-layer_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ГBА
2__inference_1-input-10-opt-mae_layer_call_fn_10734inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ГBА
2__inference_1-input-10-opt-mae_layer_call_fn_10747inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ОBЛ
2__inference_1-input-10-opt-mae_layer_call_fn_10676input-layer_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЮBЫ
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10763inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЮBЫ
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10779inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
©B¶
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10690input-layer_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
©B¶
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10704input-layer_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
C
*0
@1
A2
B3
C4"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
@0
A1
B2
C3"
trackable_list_wrapper
њ2Љє
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
‘B—
#__inference_signature_wrapper_10721input-layer_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_input-layer_layer_call_fn_10788inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_input-layer_layer_call_and_return_conditional_losses_10798inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЁ
,__inference_output-layer_layer_call_fn_10807inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ыBш
G__inference_output-layer_layer_call_and_return_conditional_losses_10817inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
N
D	variables
E	keras_api
	Ftotal
	Gcount"
_tf_keras_metric
^
H	variables
I	keras_api
	Jtotal
	Kcount
L
_fn_kwargs"
_tf_keras_metric
(:&
2SGD/m/input-layer/kernel
": 
2SGD/m/input-layer/bias
):'
2SGD/m/output-layer/kernel
#:!2SGD/m/output-layer/bias
.
F0
G1"
trackable_list_wrapper
-
D	variables"
_generic_user_object
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper…
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10690xBҐ?
8Ґ5
+К(
input-layer_input€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ …
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10704xBҐ?
8Ґ5
+К(
input-layer_input€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Њ
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10763m7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Њ
M__inference_1-input-10-opt-mae_layer_call_and_return_conditional_losses_10779m7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ £
2__inference_1-input-10-opt-mae_layer_call_fn_10603mBҐ?
8Ґ5
+К(
input-layer_input€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€£
2__inference_1-input-10-opt-mae_layer_call_fn_10676mBҐ?
8Ґ5
+К(
input-layer_input€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€Ш
2__inference_1-input-10-opt-mae_layer_call_fn_10734b7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€Ш
2__inference_1-input-10-opt-mae_layer_call_fn_10747b7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€£
 __inference__wrapped_model_10552:Ґ7
0Ґ-
+К(
input-layer_input€€€€€€€€€
™ ";™8
6
output-layer&К#
output_layer€€€€€€€€€≠
F__inference_input-layer_layer_call_and_return_conditional_losses_10798c/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€

Ъ З
+__inference_input-layer_layer_call_fn_10788X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€
Ѓ
G__inference_output-layer_layer_call_and_return_conditional_losses_10817c/Ґ,
%Ґ"
 К
inputs€€€€€€€€€

™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_output-layer_layer_call_fn_10807X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€

™ "!К
unknown€€€€€€€€€Љ
#__inference_signature_wrapper_10721ФOҐL
Ґ 
E™B
@
input-layer_input+К(
input_layer_input€€€€€€€€€";™8
6
output-layer&К#
output_layer€€€€€€€€€