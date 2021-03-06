?	?/?
*N@?/?
*N@!?/?
*N@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?/?
*N@??=?L@1x}??O9??Aj0?Gİ?Ijh??@*	?E????Y@2F
Iterator::Model?k*?¦?!????muE@)Z*oG8-??1????^?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]???Ա??!?.L? +9@)?\kF??1?7?f 5@:Preprocessing2U
Iterator::Model::ParallelMapV2??f?W??!,?W?|4@)??f?W??1,?W?|4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!	???9@)??CR%??1?????2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP?c*??!RAe?u@)P?c*??1RAe?u@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ʡE??!p&??L@)|E?^s?18z?|7C@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??m??q?!!?T???@)??m??q?1!?T???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`????!?Ҳp5Y;@)??7h?>^?1`??o???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??B+?X@Q???G/5??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??=?L@??=?L@!??=?L@      ??!       "	x}??O9??x}??O9??!x}??O9??*      ??!       2	j0?Gİ?j0?Gİ?!j0?Gİ?:	jh??@jh??@!jh??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??B+?X@y???G/5???";
sequential_27/dense_81/MatMulMatMul ^?資?! ^?資?0";
sequential_27/dense_82/MatMulMatMul ^?資?! ^?賗?0"I
+gradient_tape/sequential_27/dense_82/MatMulMatMul??]ds???!揉??ơ?0"I
+gradient_tape/sequential_27/dense_81/MatMulMatMulW?z?-???!|=h`???0"I
-gradient_tape/sequential_27/dense_82/MatMul_1MatMul??s??!'?lW?B??";
sequential_27/dense_83/MatMulMatMul????y?!)3?s??0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?+?-?y?!H??0R??"I
-gradient_tape/sequential_27/dense_83/MatMul_1MatMul?+?-?y?!?s|ɓ???"Y
8gradient_tape/sequential_27/dense_83/BiasAdd/BiasAddGradBiasAddGrad???H??u?!??^?H??"7
Nadam/Nadam/update/truedivRealDivnA.o.<r?!ҕ?Dbl??Q      Y@Yg<??x@a9?as?W@q??'?NX@y??R??|??"?
both?Your program is POTENTIALLY input-bound because 92.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 