	<???|O@<???|O@!<???|O@	2??IIӑ?2??IIӑ?!2??IIӑ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6<???|O@8L4H??M@1]????1??A?!???ɩ?I*p??a@Y?q??>s??*	??????[@2F
Iterator::Model~7ݲC???!H????H@)$??????1>'????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX???<??!?K??7@)???Y???1@B]??Y3@:Preprocessing2U
Iterator::Model::ParallelMapV2??V|C???!x???1@)??V|C???1x???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(I?L?ٖ?!d#?)4@)鹅?D???1????Cr'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef?(?7??!0p?? @)f?(?7??10p?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?cϞˬ?!?Q&?K[I@)???RAEu?1Y?Ұ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?@?Ρu?!R/g=?@)?@?Ρu?1R/g=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(~??k	??!??`? 6@)??ek}a?1???Cr???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92??IIӑ?I?ڇ
*?X@Q???80.??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8L4H??M@8L4H??M@!8L4H??M@      ??!       "	]????1??]????1??!]????1??*      ??!       2	?!???ɩ??!???ɩ?!?!???ɩ?:	*p??a@*p??a@!*p??a@B      ??!       J	?q??>s???q??>s??!?q??>s??R      ??!       Z	?q??>s???q??>s??!?q??>s??b      ??!       JGPUY2??IIӑ?b q?ڇ
*?X@y???80.??