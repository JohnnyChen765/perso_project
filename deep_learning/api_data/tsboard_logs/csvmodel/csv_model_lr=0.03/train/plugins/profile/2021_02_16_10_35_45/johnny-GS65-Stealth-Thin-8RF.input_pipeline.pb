	?-??TK@?-??TK@!?-??TK@	?j$?????j$????!?j$????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?-??TK@?)?TP?I@1??lY???A?^~?Ɍ??I??A|`g@Y???	.V??*	?l???iT@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?P??9??!?^????=@)?F? \??1?ln9@:Preprocessing2F
Iterator::ModelZ*oG8-??!\????XC@)9
3???1nu<??3@:Preprocessing2U
Iterator::Model::ParallelMapV2???{h??!L}??I?2@)???{h??1L}??I?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???n-??!0?K???6@)?[z4??17?Б\)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceD?Ac&??!*????$@)D?Ac&??1*????$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2???????!?f?N@)?߆?yu?19?weW?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'Hlwp?!s??e5@)?'Hlwp?1s??e5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???;ޔ?!??k?8@)?R]??[?15?N- @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?j$????IZ?/˳?X@Q?<̱?4??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)?TP?I@?)?TP?I@!?)?TP?I@      ??!       "	??lY?????lY???!??lY???*      ??!       2	?^~?Ɍ???^~?Ɍ??!?^~?Ɍ??:	??A|`g@??A|`g@!??A|`g@B      ??!       J	???	.V?????	.V??!???	.V??R      ??!       Z	???	.V?????	.V??!???	.V??b      ??!       JGPUY?j$????b qZ?/˳?X@y?<̱?4??