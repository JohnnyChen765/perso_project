?	????c@????c@!????c@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????c@ϼv?b@1$??:?@Ao?
????I|?q'@*	?~j?tO@2F
Iterator::Model^??6S!??!2s?ƫG@)?*n?b??1?dAC?7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??
???!????5=@)P??????1???T??7@:Preprocessing2U
Iterator::Model::ParallelMapV2?????ߍ?!??-Jx7@)?????ߍ?1??-Jx7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?o??~?!?????'@)?o??~?1?????'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9CqǛ???!i?~2@)?-????o?1D?]<?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Ss????!͌?U9TJ@)?V???l?1???c?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorgF?N?k?!6?????@)gF?N?k?16?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI[NV?^X@Q?4?=?,@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ϼv?b@ϼv?b@!ϼv?b@      ??!       "	$??:?@$??:?@!$??:?@*      ??!       2	o?
????o?
????!o?
????:	|?q'@|?q'@!|?q'@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q[NV?^X@y?4?=?,@?":
sequential_2/dense_42/MatMulMatMul??4Dt|??!??4Dt|??0"?
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits&???֌?!m????s??"H
*gradient_tape/sequential_2/dense_42/MatMulMatMul??Y???!???y?6??0"-
IteratorGetNext/_3_Send`???hɁ?!?y??1???"3
Nadam/Nadam/update/add_2AddV2?c?J??v?!H?????"1
Nadam/Nadam/update/addAddV2??????u?!?np/??"3
Nadam/Nadam/update/add_1AddV2F??dSp?!P??%>$??"9
Nadam/Nadam/update/truediv_3RealDivF??dSp?!??o)??":
sequential_2/dense_58/BiasAddBiasAdd$??Rp?!????.??":
sequential_2/dense_62/SoftmaxSoftmax$??Rp?!tO?S?3??Q      Y@YK?`m???a?}?K??X@q?>H}=X@y78p|? ??"?

both?Your program is POTENTIALLY input-bound because 94.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 