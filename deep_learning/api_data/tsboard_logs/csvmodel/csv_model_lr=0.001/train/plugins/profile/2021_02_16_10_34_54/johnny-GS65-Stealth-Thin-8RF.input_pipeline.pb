	?B</]M@?B</]M@!?B</]M@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?B</]M@N&nĠK@1DP5z5@??A      ??IqqTn?@*	D?l???U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~??@??!?C?oA@)=?+J	???1?+XwN?=@:Preprocessing2U
Iterator::Model::ParallelMapV29?~߿y??!d?sA3@)9?~߿y??1d?sA3@:Preprocessing2F
Iterator::ModelH?`?????!}-?1?B@)W|C??u??1?Em? ]2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP?Lۿ???!?-Uv?4@)???DR??1???t?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicekE???&|?!m?k?h@)kE???&|?1m?k?h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??d#٫?!????O@)????g?r?1]d??[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?M?d?q?!jpя>@)?M?d?q?1jpя>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc??????!=?2?.6@),am???R?1?dY|!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??׬b?X@QS?	?T???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	N&nĠK@N&nĠK@!N&nĠK@      ??!       "	DP5z5@??DP5z5@??!DP5z5@??*      ??!       2	      ??      ??!      ??:	qqTn?@qqTn?@!qqTn?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??׬b?X@yS?	?T???