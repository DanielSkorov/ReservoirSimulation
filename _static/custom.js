let fullmode = Boolean(false);

function toggleFullScreen() {
  let r = document.querySelector(':root');
  let rs = getComputedStyle(r);
  if (!fullmode) {
    // console.log("The value of --pst-page-max-width is: " + rs.getPropertyValue('--pst-page-max-width'));
    // console.log("The value of --pst-page-min-width is: " + rs.getPropertyValue('--pst-page-min-width'));
    r.style.setProperty('--pst-page-max-width', 'none');
    r.style.setProperty('--pst-page-min-width', '100%');
    r.style.setProperty('--pst-div-full-width', '100%');
    r.style.setProperty('--pst-sidebar-toggle-display', 'none');
    // console.log("The value of --pst-page-max-width is: " + rs.getPropertyValue('--pst-page-max-width'));
    // console.log("The value of --pst-page-min-width is: " + rs.getPropertyValue('--pst-page-min-width'));
    fullmode = Boolean(true);
  } else {
    // console.log("The value of --pst-page-max-width is: " + rs.getPropertyValue('--pst-page-max-width'));
    // console.log("The value of --pst-page-min-width is: " + rs.getPropertyValue('--pst-page-min-width'));
    r.style.setProperty('--pst-page-max-width', '88rem');
    r.style.setProperty('--pst-page-min-width', 'none');
    r.style.setProperty('--pst-div-full-width', '136%');
    r.style.setProperty('--pst-sidebar-toggle-display', 'inline-block');
    // console.log("The value of --pst-page-max-width is: " + rs.getPropertyValue('--pst-page-max-width'));
    // console.log("The value of --pst-page-min-width is: " + rs.getPropertyValue('--pst-page-min-width'));
    fullmode = Boolean(false);
  }
}



