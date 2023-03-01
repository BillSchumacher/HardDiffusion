import 'package:flutter/material.dart';

class GuidanceScaleField extends StatelessWidget {
  const GuidanceScaleField(
      {super.key, required this.setValue, required this.value});

  final Function(double) setValue;
  final double value;

  @override
  Widget build(BuildContext context) {
    return Flexible(
      child: Column(
        children: [
          Text("Guidance Scale:"),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextFormField(
              onSaved: (newValue) => setValue(double.parse(newValue!)),
              initialValue: value.toString(),
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
