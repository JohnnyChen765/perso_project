	r?_??a@r?_??a@!r?_??a@	P?6u????P?6u????!P?6u????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6r?_??a@??X3??`@1
e??k]@A??¼Ǚ??I???UG.@Y?ơ~???*	E?????V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??]?????!۝S?D@)???8a ?1(8???A@:Preprocessing2U
Iterator::Model::ParallelMapV2F{????!B????@2@)F{????1B????@2@:Preprocessing2F
Iterator::ModelVIddY??!?^?'?~A@)?.\sG??1h?\?2?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM?<i???!?6?44@)H??0~??1???3Ҹ(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????P}?!???a^@)?????P}?1???a^@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?'G?`??!?P8l?@P@)??;Fzq?16V?ҵ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???u6?o?!?U?@)???u6?o?1?U?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9P?6u????Ih?vP?PX@Q?W???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??X3??`@??X3??`@!??X3??`@      ??!       "	
e??k]@
e??k]@!
e??k]@*      ??!       2	??¼Ǚ????¼Ǚ??!??¼Ǚ??:	???UG.@???UG.@!???UG.@B      ??!       J	?ơ~????ơ~???!?ơ~???R      ??!       Z	?ơ~????ơ~???!?ơ~???b      ??!       JGPUYP?6u????b qh?vP?PX@y?W???@