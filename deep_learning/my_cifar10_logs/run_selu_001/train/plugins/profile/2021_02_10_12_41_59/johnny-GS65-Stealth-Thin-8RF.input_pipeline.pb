	:?!?'@:?!?'@!:?!?'@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-:?!?'@????U?@1?I?>@A??ܵ??IcD?Ђ@*	
ףp=?S@2F
Iterator::Model?????!?a???D@)?	Q???1??d9?4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????f??!?????9@)??_#I??1?#e?na4@:Preprocessing2U
Iterator::Model::ParallelMapV2y;?i????!1???3@)y;?i????11???3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz???3K??!"??9??6@)]??u???1?_?z\V)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW??:???!
?Krx?M@)6sHj?d??1??y??'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?2p@KW??!o*?%s$@)?2p@KW??1o*?%s$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? :vp?!??I?ۙ@)? :vp?1??I?ۙ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?45.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP	2?P@Q??_훃@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????U?@????U?@!????U?@      ??!       "	?I?>@?I?>@!?I?>@*      ??!       2	??ܵ????ܵ??!??ܵ??:	cD?Ђ@cD?Ђ@!cD?Ђ@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP	2?P@y??_훃@@