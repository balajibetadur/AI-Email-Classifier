
if(document.getElementById("op").value==="OK")
{
    document.getElementById("cv").style.display="block";
    
    document.getElementById("cv3").style.display="none";
    document.getElementById("cv1").style.display="block";
}
var d={
    "time" : {
        "1": "short time",
        "2": "moderate time"
    },
    
    
    "parameters" : {
        "1" : ["Test Size","input","20","Enter test size. ex: 20 for 20% test set"],
        "2" : ["Grid Search","select",["No","Yes"],"Tests for different hyperparameter values (Increases training time)"],
        "3" : ["Number of Estimators", "input","100","(Number fo Trees) Suggested value is between 64 - 128 trees. Huge value may increase training time"],
        "4" : ["Maximum Iterations","input","100","Maximum number of iterations taken for the solvers to converge"],
        "5" : ["Min Samples Split","input","2","The minimum number of samples required to split an internal node"],
        "6" : ["Min Samples Leaf", "input", "1", "The minimum number of samples required to be at a leaf node."],
        "7" : ["alpha", "select",["1","0", "0.1", "0.01", "0.001", "0.0001", "0.00001"], "Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing)"],
        "8" : ["loss", "select", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "epsilon_insensitive"], "The loss function to be used" ],
        "9" : ["penalty", "select", ["l1","l2","elasticnet"], "The penalty (aka regularization term) to be used"],
        "10": ["alpha_svm", "select",["0.001","0.01","0.0001"], "Constant that multiplies the regularization term."],
        "11": ["Maximum Iteration","input","5", "The maximum number of passes over the training data (aka epochs)"]
    },


    "models" : {

        "1" : {
            "1" : ["Logistic Regression",["1","2","4"]],
            "2" : ["Decesion Tree Classifier", ["1","2","3"]],
            "3" : ["Random Forest Classifier",["1","2","3","5","6"]],   
            // # "4" : ["XGBoost",["1","2","3","4","6","7"]],   
            "4" : ["Linear Support Vector Machine",["1","2","8","9","10","11"]],   
            "5" : ["Naive Bayes Classifier",["1","2","7"]]
        },
        
        "2" : {
            "1" : ["Artificial Neural Networks",["7"]],
            "2" : ["LSTM",["7"]],
            "3" : ["BERT",["6","7"]]
        }
    }
}
var z = document.getElementById("s").value;
z = z.split("$");
console.log(z);
var d_id=0;

// name = name.split("$")  
console.log(z[0]);
let i,rmbutton=document.querySelectorAll(".bz");
for (i=0;i<rmbutton.length;i++){
	if (rmbutton[i].value===document.getElementById("opp").value) 
	rmbutton[i].selected = 'selected';
}
function preee(){
    if(document.getElementById("semodel").value==='101' || document.getElementById("semodel").value==='')
   { document.getElementById("err1").style.display="block";
    document.getElementById("title").focus();
    return false; 
}
  
document.getElementById("err").style.display="none";
return true; 
}
function chec(){
    let na=document.getElementById("title").value,len=0;
    console.log(na);
    len=z.length;
    
    for (let p=0;p<len;p++)
    {
        if(z[p]===na)
        {
            document.getElementById("err").style.display="block";
            document.getElementById("title").focus();
            return 0;
        }
    }
    document.getElementById("err").style.display="none";

}
let de=2;

start();
function start(){
    let ti=document.getElementById("ti").value,algo=1;
    addalgo(ti);
}
function addalgo(ti){
    let a=document.getElementById("algo");
    a.innerHTML = "";
    let len=Object.keys(d['models'][ti]).length;
 for (i=1;i<=len;i++){
    a.innerHTML+= "<option value = '"+i+"'>"+d['models'][ti][i][0]+"</option>";
 }
  len=d['models'][ti][1][1];
 addcofig(len);
}
function changealgo(){
    let len,algo,i;
    algo=document.getElementById("algo").value;
    i=document.getElementById("ti").value;
    len=d['models'][i][algo][1];
 addcofig(len);
}
function addcofig(temp){
    let a=document.getElementById("demo");
    a.innerHTML = " ";
    
   let b='';
    let len=temp.length;
    let op,i=0;
    d_id=0;
 for (i=0;i<len;i++){
     d_id+=1;
    op=d["parameters"][temp[i]];
    if (op[1]=='input')
    b+=input(op);
    else if(op[1]=='select')
    b+=select(op);
}
a.innerHTML+='<br><div class="row">'+b+'</div>';
  

}
function input(op){
    return('<br><div class="col-6"><label for="inputPassword5">'+op[0]+'</label><input type="text" id="'+d_id+'" name="'+op[0]+'" class="form-control" value="'+op[2]+'" aria-describedby="passwordHelpBlock"><small class="form-text text-muted">'+op[3]+'</small></div>');
}
function select(op){

    let l=op[2].length,k='',j=0;
    for(j=0;j<l;j++){
          k+='<option  value = "'+op[2][j]+'" >'+op[2][j]+'</option>';
       }
    return('<br><div class="col-6"> <div class=""> <label>'+op[0]+'</label><select name="'+op[0]+'" id="'+d_id+'" class="form-control">'+k+'</select></div><small  class="form-text text-muted">'+op[3]+'</small></div>');
  
}
function ti(){
 //   alert(document.getElementById('ti').value);
}
function res(){
  //  alert("sjdhjsd");
if (chec() == 0){
return false;
} 
let o,result={};

for( o=1;o<=d_id;o++){
result[document.getElementById(o).name]=document.getElementById(o).value;
}
result["title"]=document.getElementById("title").value;
result["t_ime"]=document.getElementById("ti").value;
result["algo"]=document.getElementById("algo").value;
result["path"]=document.getElementById("fi").files;
document.getElementById("rs").value=JSON.stringify(result);
console.log(result);

return true;
}

function email(lp){
    
    document.getElementById('exampleModalLong').style.display='block';
    document.getElementById('pu').innerHTML=document.getElementById(lp).innerHTML;
    lp+='s';
    document.getElementById('lklk').innerHTML=document.getElementById(lp).innerHTML;


}















