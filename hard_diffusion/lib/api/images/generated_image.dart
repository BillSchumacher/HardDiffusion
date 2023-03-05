import 'package:hard_diffusion/api/network_service.dart';
import 'package:hard_diffusion/list_page.dart';
import 'package:uuid_type/uuid_type.dart';
import 'package:flutter/foundation.dart';

class GeneratedImage {
  final Uuid taskId;
  final int id;
  final String filename;
  final String? host;
  final DateTime createdAt;
  final DateTime? generatedAt;
  final String prompt;
  final String? negativePrompt;
  final double? duration;
  final int? seed;
  final double? guidanceScale;
  final int? numInferenceSteps;
  final int height;
  final int width;
  final String? model;
  final bool? error;

  const GeneratedImage(
      {required this.taskId,
      required this.id,
      required this.filename,
      this.host,
      required this.createdAt,
      this.generatedAt,
      required this.prompt,
      this.negativePrompt,
      this.duration,
      this.seed,
      this.guidanceScale,
      this.numInferenceSteps,
      required this.height,
      required this.width,
      this.model,
      this.error});

  factory GeneratedImage.fromJson(Map<String, dynamic> json) {
    var negativePrompt = json['negative_prompt'];
    var generatedAt = json['generated_at'];
    if (generatedAt != null) {
      generatedAt = DateTime.parse(generatedAt);
    }
    var model = json['model'];
    model ??= 'CompVis/StableDiffusion-1.4';
    return GeneratedImage(
      taskId: Uuid.parse(json['task_id']),
      id: json['id'] as int,
      filename: json['filename'] as String,
      host: json['host'],
      createdAt: DateTime.parse(json['created_at']),
      generatedAt: generatedAt,
      prompt: json['prompt'] as String,
      negativePrompt: json['negative_prompt'],
      duration: json['duration'],
      seed: json['seed'],
      guidanceScale: json['guidance_scale'] as double,
      numInferenceSteps: json['num_inference_steps'] as int,
      height: json['height'] as int,
      width: json['width'] as int,
      model: model,
      error: json['error'] as bool,
    );
  }
}

Future<ListPage<GeneratedImage>> fetchPhotos(int lastPage, int pageSize) async {
  final response = await NetworkService().get(
      'http://localhost:8000/api/images/?format=json&sort[]=-created_at&&page=$lastPage&page_size=$pageSize');

  // Use the compute function to run parsePhotos in a separate isolate.
  return compute(parsePhotos, response);
}

// A function that converts a response body into a List<Photo>.
ListPage<GeneratedImage> parsePhotos(dynamic jsonResponse) {
  final images = jsonResponse["generated_images"].cast<Map<String, dynamic>>();
  final parsedImages = images.map<GeneratedImage>((json) {
    return GeneratedImage.fromJson(json);
  }).toList();
  final parsedTotalTount = jsonResponse["meta"]["total_results"] as int;
  return ListPage<GeneratedImage>(
      itemList: parsedImages, totalCount: parsedTotalTount);
}
