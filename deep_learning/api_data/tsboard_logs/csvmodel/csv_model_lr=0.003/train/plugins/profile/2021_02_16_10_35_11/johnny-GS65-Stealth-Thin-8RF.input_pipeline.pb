	?????ML@?????ML@!?????ML@	?1O+a???1O+a??!?1O+a??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????ML@?????J@1???҇.??A?eN??Ħ?Ih%???b@Y0?Qd????*	?t??R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?????!Z??g?:@)bMeQ?E??1?V???l6@:Preprocessing2U
Iterator::Model::ParallelMapV2??Cl??!E)?]R5@)??Cl??1E)?]R5@:Preprocessing2F
Iterator::Model+?`??!??? dCE@)?|a2U??1I??jj45@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?.????!??^???8@)?0DN_χ?1?5??.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/3l??{?!?һ?P&"@)/3l??{?1?һ?P&"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??X?_"??!Z???L@)Oʤ?6 k?1?f`Y	?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??u??i?!ľD???@)??u??i?1ľD???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H??_??!?z!Aes:@)???1??W?1??(l4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?1O+a??Ib%???X@Q{?%q???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????J@?????J@!?????J@      ??!       "	???҇.?????҇.??!???҇.??*      ??!       2	?eN??Ħ??eN??Ħ?!?eN??Ħ?:	h%???b@h%???b@!h%???b@B      ??!       J	0?Qd????0?Qd????!0?Qd????R      ??!       Z	0?Qd????0?Qd????!0?Qd????b      ??!       JGPUY?1O+a??b qb%???X@y{?%q???