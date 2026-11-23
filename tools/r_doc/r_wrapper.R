# r_wrapper.R — generates one R function file per BayesForge distribution,
# wrapping the Python API programmatically through reticulate.
# Dev-only generator: it lives in tools/ and is kept out of the Python package.
# The generated functions call `.BF_env$.bf_instance$dist$<name>(...)`, which
# the live R bridge defines at runtime.
#
# How to run (from the repo root):
#   BF_R_OUT=./R Rscript tools/r_doc/r_wrapper.R
#   BF_DOC_JSON  path to the roxygen JSON with the R docs
#                (default: RDoc.json next to this script)
#   BF_R_OUT     output directory for the generated .R files (default: "./R")
# Requirement: a Python environment with BayesForge importable and visible to
# reticulate; this script does not install anything.
library(magrittr)
library(reticulate)
library(jsonlite)
inspect <- import("inspect")
BayesForge <- import("BayesForge")
.bf_instance <- BayesForge$bf(print_devices_found = FALSE)
dist <- .bf_instance$dist
script_dir <- function() {
  # Directory of this script so the default doc JSON does not depend on the CWD.
  args <- commandArgs(trailingOnly = FALSE)
  entry <- grep("^--file=", args, value = TRUE)
  if (length(entry) > 0) return(dirname(sub("^--file=", "", entry[[1]])))
  "tools/r_doc"  # fallback: sourced from the repo root
}
dirout_path <- Sys.getenv("BF_R_OUT", "./R")
doc_json <- Sys.getenv("BF_DOC_JSON", file.path(script_dir(), "RDoc.json"))
if (!file.exists(doc_json)) {
  stop(sprintf(
    "roxygen doc JSON not found: %s. Regenerate it with bf_doc.py + the LLM step + fill_params.py (see tools/r_doc/README.md), or set BF_DOC_JSON.",
    doc_json))
}
data <- fromJSON(doc_json)
strip_fence <- function(doc) {
  doc <- sub("^\\s*```[A-Za-z]*\\s*\\n", "", doc)
  sub("\\n?\\s*```\\s*$", "", doc)
}

extract_values <- function(param) {
  # Convert to string and extract the text within quotes
  as.character(param) %>%
    sub(".*?\"(.*?)\".*", "\\1", .)
}


build_function = function(py_name,
                          func_name,
                          name_file,
                          signature,
                          output_dir=dirout_path,
                          docString = ""){
  # Create directory if it doesn't exist
  dir.create(output_dir, showWarnings = FALSE)

  params <- signature$parameters
  param_names <- reticulate:::py_dict_get_keys_as_str(params)
  named_list <- as.list(params$copy())
  extracted_terms <- lapply(named_list, extract_values)
  extracted_terms <- extracted_terms[!names(extracted_terms) %in% c("self")]
  extracted_terms <- extracted_terms[!names(extracted_terms) %in% c("args")]
  extracted_terms <- lapply(extracted_terms, function(x) gsub("\\(", "c(", gsub("\\]", ")", x)))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("\\[", "c(", gsub("\\]", ")", x)))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("None", "py_none()", x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("True", paste(T), x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("False", paste(F), x))
  extracted_terms <- lapply(extracted_terms, function(x) gsub("<function\\s+(\\w+)\\s+at\\s+0x[0-9A-Fa-f]+>", "'numpyro.\\1'", x))
  default_params <- paste(extracted_terms, collapse = ", ")


  if('kwargs' %in% names(extracted_terms)){
    default_paramsR = gsub("\\*\\*kwargs", '...', default_params)
    default_paramsP = gsub("\\*\\*kwargs", 'list(...)', default_params)
    shape_inside=FALSE

    if(grepl('shape',default_params)){
      shape_inside=TRUE
    }

  }else{
    shape_inside=FALSE
    if(grepl('shape',default_params)){
      shape_inside=TRUE
    }

    default_paramsR = default_params
    tmp=strsplit(default_params,',')
    for (a in 1:length(tmp[[1]])) {
      tmp[[1]][a]=gsub("([^=]+)=.*", "\\1=\\1", tmp[[1]][a])
    }
    tmp=paste(unlist(tmp),collapse = ', ')

    default_paramsP = tmp
  }

  # Generate the new R function dynamically
  if(shape_inside){

    func_body <- paste0(func_name,
                        "=function(", paste(default_paramsR), ") { \n",
                        #"     BF=importbi(platform='cpu')\n",
                        "     shape=do.call(tuple, as.list(as.integer(shape)))\n",
                        "     seed=as.integer(seed);\n",
                        "     .BF_env$.bf_instance$dist$", py_name, "(",paste(default_paramsP), ")\n",
                        "}")

    func_body = gsub("loc: jax.Array", "loc=0.0", func_body)
    func_body = gsub("covariance_row:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_row=None,", func_body)
    func_body = gsub("covariance_row: jax.Array = covariance_row: jax.Array ,", "covariance_row = covariance_row,", func_body)
    func_body = gsub("covariance_rfft:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_rfft=None,", func_body)
    func_body = gsub(" covariance_rfft: jax.Array = covariance_rfft: jax.Array ,", "covariance_rfft=covariance_rfft,", func_body)

  }else{
    func_body <- paste0("function(", paste(default_paramsR), ") {",
                      "    .BF_env$.bf_instance$dist$", py_name, "(",
                      paste(default_paramsP), ")",
                      "}")
    func_body = gsub("loc: jax.Array", "loc=0.0", func_body)
    func_body = gsub("covariance_row:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_row=None,", func_body)
    func_body = gsub("covariance_rfft:\\s*jax\\.Array\\s*=\\s*py_none\\(\\),?", "covariance_rfft=None,", func_body)
  }


  # Assign the function as before
  eval(parse(text = func_body))
  assign(func_name, eval(parse(text = func_body)))

  # Write the function to a file in the specified directory
  # Construct the full file path
  file_path <- file.path(output_dir, paste0(name_file, ".R"))
  print(file_path)

  # Write the function to the file
  file_con <- file(file_path, "w")
  writeLines(docString, file_con)
  writeLines(func_body, file_con)
  close(file_con)
}


# Call distributions----------------------
attrs <- py_list_attributes(dist)
no=c("__class__",
     "__delattr__",
     "__dict__",
     "__dir__",
     "__doc__",
     "__eq__",
     "__format__",
     "__ge__",
     "__getattribute__",
     "__getstate__",
     "__gt__",
     "__hash__",
     "__init__",
     "__init_subclass__",
     "__le__",
     "__lt__",
     "__module__",
     "__ne__",
     "__new__",
     "__reduce__",
     "__reduce_ex__",
     "__repr__",
     "__setattr__",
     "__sizeof__",
     "__str__",
     "__subclasshook__",
     "__weakref__",
     "sineskewed")
attrs <- attrs[!startsWith(attrs, "_") & !attrs %in% no]

for (a in attrs){
  obj <- tryCatch(reticulate::py_get_attr(dist, a), error = function(e) NULL)
  if (is.null(obj) || !reticulate::py_callable(obj)) next
  func_name <- paste0("bf.dist.", a)
  doc <- data[[a]]
  if (is.null(doc)) doc <- ""
  doc <- strip_fence(doc)
  build_function(py_name = a,
                 name_file = a,
                 func_name = func_name,
                 signature = inspect$signature(obj),
                 docString = doc)
}




