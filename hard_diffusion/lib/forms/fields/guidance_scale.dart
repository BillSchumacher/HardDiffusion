import 'package:flutter/material.dart';

class GuidanceScaleField extends StatelessWidget {
  const GuidanceScaleField({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text("Guidance Scale:"),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white,
                hintText: '',
              ),
            ),
          ),
        ],
      ),
    );
  }
}
