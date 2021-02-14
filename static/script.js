function disableConvert() {
  $("#convertBtn").prop("disabled", true);
  $("#convertBtn").addClass("disabled");
}
function enableConvert() {
  $("#convertBtn").prop("disabled", false);
  $("#convertBtn").removeClass("disabled");
}
function showUploadedImage(input) {
  if (input.files && input.files[0]) {
    var output = document.getElementById("uploadedImage");
    output.src = URL.createObjectURL(input.files[0]);
    output.onload = function () {
      URL.revokeObjectURL(output.src);
      enableConvert();
    };
  }
}
async function sanitizeAndConvert() {
  var formData = new FormData();
  var fileUpload = document.getElementById("fileUpload");
  formData.append("image", fileUpload.files[0]);
  $("#convertBtn").html(`<i class="fa fa-spinner fa-spin"></i> Converting`);
  var response = await fetch("/convert", {
    method: "POST",
    body: formData,
  });
  disableConvert();
  return response.json();
}

$(document).ready(function () {
  disableConvert();
  $("#fileUpload").change(function () {
    showUploadedImage(this);
  });
  $("#convert_form").on("submit", function (e) {
    e.preventDefault();
    sanitizeAndConvert().then((data) => {
      var output = document.getElementById("uploadedImage");
      output.src = data["image_link"];
      output.onload = function () {
        enableConvert();
        $("#convertBtn").html(`Convert!`);
      };
    });
    return false;
  });
});
