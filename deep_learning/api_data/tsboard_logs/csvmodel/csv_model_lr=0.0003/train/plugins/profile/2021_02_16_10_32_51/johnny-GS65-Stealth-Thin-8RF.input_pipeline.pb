	4K??zJ@4K??zJ@!4K??zJ@	8? ?#??8? ?#??!8? ?#??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails64K??zJ@Y??9??H@1G仔?d??A?n?1??I
?5??@YE?J?E??*	cX9??S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???D???!x??V1'<@)??F???1H_??07@:Preprocessing2F
Iterator::Model??V'g??!??z{YD@)??p?Ws??1??},L,4@:Preprocessing2U
Iterator::Model::ParallelMapV2dZ???Z??!tTx?f4@)dZ???Z??1tTx?f4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1?t?????!4?9%?E7@)n½2oՅ?1%?:?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??+,???!E?]???#@)??+,???1E?]???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?h?^??!P????M@)5D?ov?1?
????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3???/p?!??????@)3???/p?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapaũ??,??!}'u?8@)??Os?"S?1?T7}~w??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no98? ?#??I??<?̏X@Q??Ϡ6???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Y??9??H@Y??9??H@!Y??9??H@      ??!       "	G仔?d??G仔?d??!G仔?d??*      ??!       2	?n?1???n?1??!?n?1??:	
?5??@
?5??@!
?5??@B      ??!       J	E?J?E??E?J?E??!E?J?E??R      ??!       Z	E?J?E??E?J?E??!E?J?E??b      ??!       JGPUY8? ?#??b q??<?̏X@y??Ϡ6???