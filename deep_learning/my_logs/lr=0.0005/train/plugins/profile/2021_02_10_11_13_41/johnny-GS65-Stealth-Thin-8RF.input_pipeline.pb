	I?s
??b@I?s
??b@!I?s
??b@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-I?s
??b@?b?0m|a@1x? #?@A????C§?I?(???@*	?G?z&X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???O???!???/:?@)V}??b??1~??m?9@:Preprocessing2U
Iterator::Model::ParallelMapV2(?N>=??!??2??{6@)(?N>=??1??2??{6@:Preprocessing2F
Iterator::ModelްmQf???!8?<?D@)?9?S?ɒ?1?U݇e?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?G?`็?!nm????'@)?G?`็?1nm????'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?,??o??!!???Y?4@)??Z	?%??1Դ^?V!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Z|
????!??w??BM@)#??fF?z?1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJΉ=??u?!8?v?X?@)JΉ=??u?18?v?X?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??YAYX@Q?h????@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b?0m|a@?b?0m|a@!?b?0m|a@      ??!       "	x? #?@x? #?@!x? #?@*      ??!       2	????C§?????C§?!????C§?:	?(???@?(???@!?(???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??YAYX@y?h????@