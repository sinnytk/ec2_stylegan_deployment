var imageUploaded = false;
var convertBtn = $("#convertBtn");
var fileInput = $("#fileInput");

function validateImageUpload(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#uploadedImage").attr("src", e.target.result);
      imageUploaded = true;
    };
    reader.readAsDataURL(input.files[0]);
  }
}
function convertImage(btn) {
  $(btn).prop("disabled", true);
  $(btn).html(`<i class="fa fa-spinner fa-spin"></i> Converting`);
  $(fileInput).prop("disabled", true);
  $(fileInput).addClass("disabled");
}

$(document).ready(function () {
  convertBtn.prop("disabled", true);

  fileInput.change(function () {
    validateImageUpload(this);
    convertBtn.prop("disabled", false);
    convertBtn.removeClass("disabled");
  });
  convertBtn.click(function () {
    convertImage(this);
  });
});
