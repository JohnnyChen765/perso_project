	?g\WU`@?g\WU`@!?g\WU`@	o?!&??o?!&??!o?!&??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?g\WU`@?=&R?D^@1???G?@AS?
cA??I?A|`?@Y(?N>=???*	??~j??R@2F
Iterator::Model?"M?<??!??D?tF@)?GĔH??1?b=\??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat |(ђǓ?!???[??9@)!>???@??1???DT-5@:Preprocessing2U
Iterator::Model::ParallelMapV2??R?r/??!???,?5@)??R?r/??1???,?5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?U?????!"k?%;?8@) ??*Q???1?=XD?-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice6Y???}?!??w?1m#@)6Y???}?1??w?1m#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip!;oc?#??!N?p?G?K@)?"??l?1z?֗?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???k?6l?!jɈ]ha@)???k?6l?1jɈ]ha@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9n?!&??IU??P:X@Q?,?ҌI@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?=&R?D^@?=&R?D^@!?=&R?D^@      ??!       "	???G?@???G?@!???G?@*      ??!       2	S?
cA??S?
cA??!S?
cA??:	?A|`?@?A|`?@!?A|`?@B      ??!       J	(?N>=???(?N>=???!(?N>=???R      ??!       Z	(?N>=???(?N>=???!(?N>=???b      ??!       JGPUYn?!&??b qU??P:X@y?,?ҌI@