	?KToL@?KToL@!?KToL@	U??,?_??U??,?_??!U??,?_??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?KToL@??SDJ@1??P?????A)=?K?e??I??[???@YϞ??$x??*	?K7?A0S@2U
Iterator::Model::ParallelMapV2????je??!?(A?g7@)????je??1?(A?g7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat~?*O ???!LB???:@)?@???1?]?O?5@:Preprocessing2F
Iterator::Modelf??E???!?EKEiF@)???Ր?1mbU ?j5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)&o?????!?3R?M?5@)?!??gx??1???h??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceTUh ??|?!??A?R"@)TUh ??|?1??A?R"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!n?????K@)??Gߤip?1L\?]??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?f??o?!贐K?@);?f??o?1贐K?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????W??!Z??>V7@)??;??~V?1/j&???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U??,?_??I?.???X@Q??V_ɬ??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??SDJ@??SDJ@!??SDJ@      ??!       "	??P???????P?????!??P?????*      ??!       2	)=?K?e??)=?K?e??!)=?K?e??:	??[???@??[???@!??[???@B      ??!       J	Ϟ??$x??Ϟ??$x??!Ϟ??$x??R      ??!       Z	Ϟ??$x??Ϟ??$x??!Ϟ??$x??b      ??!       JGPUYU??,?_??b q?.???X@y??V_ɬ??