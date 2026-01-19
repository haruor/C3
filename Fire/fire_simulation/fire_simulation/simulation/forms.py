from django import forms


class UploadCSVForm(forms.Form):
    elevation_csv = forms.FileField(label="Elevation CSV")
    vegetation_csv = forms.FileField(label="Vegetation CSV")
    ignition_csv = forms.FileField(label="Ignition CSV（任意）", required=False)

    def clean_elevation_csv(self):
        f = self.cleaned_data["elevation_csv"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("CSVファイルを選択してください。")
        return f

    def clean_vegetation_csv(self):
        f = self.cleaned_data["vegetation_csv"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("CSVファイルを選択してください。")
        return f

    def clean_ignition_csv(self):
        f = self.cleaned_data.get("ignition_csv")
        if not f:
            return None
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("CSVファイルを選択してください。")
        return f