	a???`@a???`@!a???`@	u???????u???????!u???????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6a???`@]?@??_@1_???݀@A?l??爤?Ih?????@Y?"R?.??*	R???Ky@2F
Iterator::Model????]M??!?1??A}V@)??^??1?e?֖U@:Preprocessing2U
Iterator::Model::ParallelMapV2o???׍?!?{٦e?@)o???׍?1?{٦e?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??W??͔?!8?t?@)?1>?^???1??c?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?^ ??!?^N??h@)?J?*n??1s?.<Bp@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice??? !?w?!???????)??? !?w?1???????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}?K??Ϥ?!?s???$@)k*??.?n?1@z??oy??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Y?rLf?!?IM???)??Y?rLf?1?IM???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 94.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9t???????I???\EX@Q?uE???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?@??_@]?@??_@!]?@??_@      ??!       "	_???݀@_???݀@!_???݀@*      ??!       2	?l??爤??l??爤?!?l??爤?:	h?????@h?????@!h?????@B      ??!       J	?"R?.???"R?.??!?"R?.??R      ??!       Z	?"R?.???"R?.??!?"R?.??b      ??!       JGPUYt???????b q???\EX@y?uE???@