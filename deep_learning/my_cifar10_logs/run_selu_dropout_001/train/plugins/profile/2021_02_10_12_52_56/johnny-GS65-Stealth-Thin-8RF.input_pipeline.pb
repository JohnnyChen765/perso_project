	????c@????c@!????c@      ??!       "n
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
????:	|?q'@|?q'@!|?q'@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q[NV?^X@y?4?=?,@