	vS?k%$M@vS?k%$M@!vS?k%$M@	?#?IH???#?IH??!?#?IH??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6vS?k%$M@U2 Tq)K@1?,'?????A??2nj??I???@Y??t?(%??*	     hS@2F
Iterator::Model????g???!ozӛ??F@)$)?ahu??1???r?87@:Preprocessing2U
Iterator::Model::ParallelMapV2??·g	??!]Z????6@)??·g	??1]Z????6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS?r/0+??!^?P??_9@)>???4`??1?"?4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????ב?!??X-?r6@)Sz??˄?1?X??(*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Ң>?}?!,MX?y?"@)??Ң>?}?1,MX?y?"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Wn?!??-@)????Wn?1??-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Y????!??,d!K@)?3?9A?l?1 ????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?8Q????!|??578@)H?SȕzV?17?߇?G??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?#?IH??I.D????X@Q1??ǽ~??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U2 Tq)K@U2 Tq)K@!U2 Tq)K@      ??!       "	?,'??????,'?????!?,'?????*      ??!       2	??2nj????2nj??!??2nj??:	???@???@!???@B      ??!       J	??t?(%????t?(%??!??t?(%??R      ??!       Z	??t?(%????t?(%??!??t?(%??b      ??!       JGPUY?#?IH??b q.D????X@y1??ǽ~??