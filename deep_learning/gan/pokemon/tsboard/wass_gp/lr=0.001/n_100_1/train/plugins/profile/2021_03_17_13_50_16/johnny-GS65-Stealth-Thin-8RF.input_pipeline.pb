	5?? %?@5?? %?@!5?? %?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-5?? %?@????Gjn@1?*?b?z@A?C9Ѯ??I??1=aI@*	?Q??r?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV20e???.??!??-aD#X@)	?Į????1??c?V@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip j?!?
??!>???@)?ᱟ?R??1?UndG)@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle FCƣT?!?????@)FCƣT?1?????@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Ũk???!MGq?K)??)??Ũk???1MGq?K)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Md????!|???@)?$?@??1???????:Preprocessing2F
Iterator::ModelmY?.???!??O?s?@))_?BFg?1f??V?;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Xz:ӊB@Q=???,uO@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Gjn@????Gjn@!????Gjn@      ??!       "	?*?b?z@?*?b?z@!?*?b?z@*      ??!       2	?C9Ѯ???C9Ѯ??!?C9Ѯ??:	??1=aI@??1=aI@!??1=aI@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Xz:ӊB@y=???,uO@