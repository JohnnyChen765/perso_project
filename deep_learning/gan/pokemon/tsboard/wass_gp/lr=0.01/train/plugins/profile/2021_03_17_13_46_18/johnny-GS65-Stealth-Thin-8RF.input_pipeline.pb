	4??K?@4??K?@!4??K?@	?ߟ?]?v??ߟ?]?v?!?ߟ?]?v?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails64??K?@??%VF?m@1?Gߤ?.{@A?1%????I?k?,	H @Y?H?5??*	?z?Gn?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2:?,B???!??!B?X@)S@?? k??1VؙS{V@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle Ș?????!?G?#?V@)Ș?????1?G?#?V@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip u?(%???!??z??@)Z+??6??1L??)?? @:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??뉮??!???G????)??뉮??1???G????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismu?)?:??!ɲ?(?@)>?h??i??1?t2
?e??:Preprocessing2F
Iterator::Model?]M?????!?ϻwa@)\ ?K?b?13????<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?ߟ?]?v?I?Aۯ'B@Q?6?#?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??%VF?m@??%VF?m@!??%VF?m@      ??!       "	?Gߤ?.{@?Gߤ?.{@!?Gߤ?.{@*      ??!       2	?1%?????1%????!?1%????:	?k?,	H @?k?,	H @!?k?,	H @B      ??!       J	?H?5???H?5??!?H?5??R      ??!       Z	?H?5???H?5??!?H?5??b      ??!       JGPUY?ߟ?]?v?b q?Aۯ'B@y?6?#?O@