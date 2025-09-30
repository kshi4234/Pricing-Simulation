function openTab(event, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName('tablinks');
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace("active", "");
    }
    document.getElementById(tabName).style.display = "block";
    event.currentTarget.className += "active";
}


window.addEventListener("DOMContentLoaded", function() {
    let my_form = this.document.getElementById('my-form')
    my_form.addEventListener("submit", function(e) {
        e.preventDefault(); // before the code
        /* do what you want with the form */
        fetch(my_form.action, {
            method: "post",
            body: new FormData(my_form)
        })
        // Will be triggered on form submit
        console.log('hi');
    })
});